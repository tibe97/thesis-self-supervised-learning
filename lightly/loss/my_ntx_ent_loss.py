""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from numpy import add
import torch
from torch import nn
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
import ipdb

class MyNTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like 
    the one described in the MoCo[1] paper.

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    [1] MoCo, 2020, https://arxiv.org/abs/1911.05722
    
    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank. 
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.

    Raises:
        ValueError if abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:

        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

    def __init__(self,
                 temperature: float = 0.2,
                 num_negatives: int = 256,
                 memory_bank_size: int = 0,
                 add_swav_loss: bool = False):
        super(MyNTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8
        self.num_negatives = num_negatives
        self.softmax = nn.Softmax(dim=1)
        self.add_swav_loss = add_swav_loss

        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor,
                q0_assign: torch.Tensor,
                q1: torch.Tensor,
                negatives: torch.Tensor):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as 
        negative samples.

            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Shape: (batch_size, embedding_size)
                out1:
                    Output projections of the second set of transformed images.
                    Shape: (batch_size, embedding_size)
                q0_assign: 
                    Cluster assignments of the original samples used to compute nearest neighbors
                    Used for SwAV loss (optional)
                q1:
                    Predicted cluster assignement directly taken from the output of the prototype 
                    layer of the network.
                    Used for SwAV loss (optional)
                sim_negatives: 
                    Computed similarities between the nearest neighbors and the negatives
                    sampled with hard negative mining. We just return the similarities because 
                    it's all we need to compute the loss.

            Returns:
                Contrastive Cross Entropy Loss value.

        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)


        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # use negatives from memory bank
            #negatives = torch.nn.functional.normalize(negatives, dim=1)
            #ipdb.set_trace()
            negatives = torch.transpose(negatives, 1, 2).to(device) # transpose to (batch_size, embedding_size, num_negatives)
            
            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum('nc,nc->n', out0, out1).unsqueeze(-1).to(device)

            # Compute sim_neg with negatives. Problem: for each positive there are different negatives.
            # We can't use the same einsum. We can use batch matrix multiplication einsum:
            # torch.einsum('i1c,icm->i1m', [a, b])
            # Each positive becomes a sample indexed along "i", while the negatives for the i-th positive
            # are stacked in a matrix at the i-th index. At the end we have to reshape the result into a vector
            # We also have to prepare the tensor of negatives accordingly
            #sim_neg = torch.einsum('nc,ck->nk', out0, negatives)
            # n1c, ncm -> n1m
            sim_neg = torch.einsum('nzc,ncm->nzm', torch.transpose(torch.unsqueeze(out0, 0), 0, 1), negatives)
            sim_neg = torch.squeeze(sim_neg, 1)

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)

        else:
            # use other samples from batch as negatives
            output = torch.cat((out0, out1), axis=0).to(device)

            # the logits are the similarity matrix divided by the temperature
            logits = torch.einsum('nc,mc->nm', output, output) / self.temperature
            # We need to removed the similarities of samples to themselves
            logits = logits[~torch.eye(2*batch_size, dtype=torch.bool, device=out0.device)].view(2*batch_size, -1)

            # The labels point from a sample in out_i to its equivalent in out_(1-i)
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            labels = torch.cat([labels + batch_size - 1, labels])
        

        contrastive_loss = self.cross_entropy(logits, labels)
        loss = contrastive_loss
        swav_loss = None
        
        alpha = torch.tensor(0.0) # swav_loss influence on the overall loss
        if self.add_swav_loss: # and negatives is not None: 
            p1 = self.softmax(q1 / self.temperature)
            swav_loss = - torch.mean(torch.sum(q0_assign * torch.log(p1), dim=1))
            loss += alpha * swav_loss
            
        return loss, swav_loss, contrastive_loss


