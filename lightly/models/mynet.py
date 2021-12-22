""" Basically NNCLR Model with prototype head for cluster prediction"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn


def _prediction_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int) -> nn.Sequential:
    """Prediction MLP. The original paper's implementation has 2 layers, with
    BN applied to its hidden fc layers but no BN or ReLU on the output fc layer.

    Note that the hidden dimensions should be smaller than the input/output
    dimensions (bottleneck structure). The default implementation using a
    ResNet50 backbone has an input dimension of 2048, hidden dimension of 512,
    and output dimension of 2048

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims:
            Hidden dimension of all the fully connected layers (should be a
            bottleneck!)
        out_dims:
            Output Dimension of the final linear layer.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Linear(h_dims, out_dims)

    prediction = nn.Sequential(l1, l2)
    return prediction


def _projection_mlp(num_ftrs: int,
                    h_dims: int,
                    out_dim: int,
                    num_layers: int = 3) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with
    BN applied to its hidden fc layers but no ReLU on the output fc layer.
    The CIFAR-10 study used a MLP with only two layers.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims:
            Hidden dimension of all the fully connected layers.
        out_dims:
            Output Dimension of the final linear layer.
        num_layers:
            Controls the number of layers; must be 2 or 3. Defaults to 3.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(num_ftrs, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dim),
                       nn.BatchNorm1d(out_dim))

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError(
            "Only MLPs with 2 and 3 layers are implemented.")

    return projection


def _prototype_layer(proj_dim: int,
                    nmb_prototypes: int) -> nn.Linear:
    """Prototype linear layer from SwAV method. This layer does not contain any non-linearity. 
    This layer is the prototype layer and is denoted by C. 
    This layer maps Z to K trainable prototype vectors. The output of the layer is the dot product 
    between Z and the prototypes. 
    The associated “weights” matrix (the one updated during backpropagation) of this layer can be 
    considered a learnable prototype bank.

    We use this prototype layer to get cluster assignments when performing positive and negative mining."""
    prototypes = None
    if nmb_prototypes > 0:
        prototypes = nn.Linear(proj_dim, nmb_prototypes, bias=False)   
    else:
        raise NotImplementedError(
            "The number of prototypes must be greater than 0.")

    return prototypes


class MyNet(nn.Module):
    """Implementation of the NNCLR[0] architecture

    Recommended loss: :py:class:`lightly.loss.ntx_ent_loss.NTXentLoss`
    Recommended module: :py:class:`lightly.models.modules.nn_memory_bank.NNmemoryBankModule`

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim: 
            Dimension of the hidden layer of the projection head.
        pred_hidden_dim:
            Dimension of the hidden layer of the predicion head.
        out_dim:
            Dimension of the output (after the projection head).
        num_mlp_layers:
            Number of linear layers for MLP.

    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>> 
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))

    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 512,
                 proj_hidden_dim: int = 2048,
                 pred_hidden_dim: int = 4096,
                 nmb_prototypes: int = 3000,
                 out_dim: int = 256,
                 num_mlp_layers: int = 3,
                 normalize=True):

        super(MyNet, self).__init__()

        self.l2norm = normalize # normalize projection and prediction outputs
        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = \
            _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)
        
        self.prediction_mlp = \
            _prediction_mlp(out_dim, pred_hidden_dim, out_dim)
        
        # input of prototype layer is output from projection mlp
        self.prototypes_layer = \
            _prototype_layer(out_dim, nmb_prototypes)

        # Initialization from SwAV implementation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
        """Embeds and projects the input images.

        Extracts features with the backbone and applies the projection
        head to the output space. If both x0 and x1 are not None, both will be
        passed through the backbone and projection head. If x1 is None, only
        x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output projection of x0 and (if x1 is not None) the output
            projection of x1. If return_features is True, the output for each x
            is a tuple (out, f) where f are the features before the projection
            head.

        Examples:
            >>> # output is a tuple
            >>> out = (projection, prediction, prototypes)
            >>>
            >>> # single input, single output
            >>> out = model(x) 
            >>> 
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)

        """
        
        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_mlp(f0)
        p0 = self.prediction_mlp(z0)

        if self.l2norm:
            z0 = nn.functional.normalize(z0, dim=1, p=2)
            p0 = nn.functional.normalize(p0, dim=1, p=2)
        z0_swav = z0.detach().clone()
        q0 = self.prototypes_layer(z0_swav)

        out0 = (z0, p0, q0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        # return out0 if x1 is None
        if x1 is None:
            return out0

        # forward pass of second input x1
        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_mlp(f1)
        p1 = self.prediction_mlp(z1)

        if self.l2norm:
            z1 = nn.functional.normalize(z1, dim=1, p=2)
            p1 = nn.functional.normalize(p1, dim=1, p=2)
        
        z1_swav = z1.detach().clone()
        q1 = self.prototypes_layer(z1_swav)

        out1 = (z1, p1, q1)

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        # return both outputs
        return out0, out1

        
    def on_after_backward(self):
        for name, p in self.named_parameters():
            if "prototypes_layer" in name:
                p.grad = None
