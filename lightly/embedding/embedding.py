""" Embedding Strategies """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import time

import torch
import lightly
from lightly.embedding._base import BaseEmbedding
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
from tqdm import tqdm
from datetime import datetime

if lightly._is_prefetch_generator_available():
    from prefetch_generator import BackgroundGenerator


class SelfSupervisedEmbedding(BaseEmbedding):
    """Implementation of self-supervised embedding models.

    Implements an embedding strategy based on self-supervised learning. A
    model backbone, self-supervised criterion, optimizer, and dataloader are
    passed to the constructor. The embedding itself is a pytorch-lightning
    module.

    The implementation is based on contrastive learning.

    * SimCLR: https://arxiv.org/abs/2002.05709
    * MoCo: https://arxiv.org/abs/1911.05722
    * SimSiam: https://arxiv.org/abs/2011.10566

    Attributes:
        model:
            A backbone convolutional network with a projection head.
        criterion:
            A contrastive loss function.
        optimizer:
            A PyTorch optimizer.
        dataloader:
            A torchvision dataloader.
        scheduler:
            A PyTorch learning rate scheduler.

    Examples:
        >>> # define a model, criterion, optimizer, and dataloader above
        >>> import lightly.embedding as embedding
        >>> encoder = SelfSupervisedEmbedding(
        >>>     model,
        >>>     criterion,
        >>>     optimizer,
        >>>     dataloader,
        >>> )
        >>> # train the self-supervised embedding with default settings
        >>> encoder.train_embedding()
        >>> # pass pytorch-lightning trainer arguments as kwargs
        >>> encoder.train_embedding(max_epochs=10)

    """

    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: torch.utils.data.DataLoader,
                 scheduler=None):

        super(SelfSupervisedEmbedding, self).__init__(
            model, criterion, optimizer, dataloader, scheduler)

    def embed(self,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device = None,
              to_numpy: bool = True):
        """Embeds images in a vector space.

        Args:
            dataloader:
                A PyTorch dataloader.
            device:
                Selected device (`cpu`, `cuda`, see PyTorch documentation)
            to_numpy:
                Whether to return the embeddings as numpy array.

        Returns:
            A tuple consisting of a tensor or ndarray of embeddings
            with shape n_images x num_ftrs and labels, fnames

        Examples:
            >>> # embed images in vector space
            >>> embeddings, labels, fnames = encoder.embed(dataloader)

        """

        self.model.eval()
        embeddings, labels, fnames = None, None, []

        if lightly._is_prefetch_generator_available():
            pbar = tqdm(BackgroundGenerator(dataloader, max_prefetch=3),
                        total=len(dataloader))
        else:
            pbar = tqdm(dataloader, total=len(dataloader))

        efficiency = 0.
        embeddings = []
        labels = []
        with torch.no_grad():

            start_time = time.time()
            for (img, label, fname) in pbar:

                img = img.to(device)
                label = label.to(device)

                fnames += [*fname]

                batch_size = img.shape[0]
                prepare_time = time.time()

                emb = self.model.backbone(img)
                emb = emb.detach().reshape(batch_size, -1)

                embeddings.append(emb)
                labels.append(label)

                process_time = time.time()

                efficiency = \
                    (process_time - prepare_time) / (process_time - start_time)
                pbar.set_description(
                    "Compute efficiency: {:.2f}".format(efficiency))
                start_time = time.time()

            embeddings = torch.cat(embeddings, 0)
            labels = torch.cat(labels, 0)
            if to_numpy:
                embeddings = embeddings.cpu().numpy()
                labels = labels.cpu().numpy()

        return embeddings, labels, fnames

class NNCLRSelfSupervisedEmbedding(BaseEmbedding):
    """Implementation of self-supervised embedding models.

    Adapted implementation for NNCLR method

    The implementation is based on contrastive learning.


    Attributes:
        model:
            A backbone convolutional network with a projection head.
        criterion:
            A contrastive loss function.
        optimizer:
            A PyTorch optimizer.
        dataloader:
            A torchvision dataloader.
        scheduler:
            A PyTorch learning rate scheduler.

    Examples:
        >>> # define a model, criterion, optimizer, and dataloader above
        >>> import lightly.embedding as embedding
        >>> encoder = SelfSupervisedEmbedding(
        >>>     model,
        >>>     criterion,
        >>>     optimizer,
        >>>     dataloader,
        >>> )
        >>> # train the self-supervised embedding with default settings
        >>> encoder.train_embedding()
        >>> # pass pytorch-lightning trainer arguments as kwargs
        >>> encoder.train_embedding(max_epochs=10)

    """

    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: torch.utils.data.DataLoader,
                 nn_replacer: MyNNMemoryBankModule,
                 warmup_epochs: int = 0,
                 scheduler=None):

        super(NNCLRSelfSupervisedEmbedding, self).__init__(
            model, criterion, optimizer, dataloader, scheduler)
        
        dirpath="checkpoints/" + datetime.today().strftime('%Y-%m-%d')
        super(NNCLRSelfSupervisedEmbedding, self).init_checkpoint_callback(
            save_last=True, save_top_k=1, monitor='loss', dirpath=dirpath)
        self.nn_replacer = nn_replacer
        self.warmup_epochs = warmup_epochs

    def embed(self,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device = None,
              to_numpy: bool = True):
        """Embeds images in a vector space.

        Args:
            dataloader:
                A PyTorch dataloader.
            device:
                Selected device (`cpu`, `cuda`, see PyTorch documentation)
            to_numpy:
                Whether to return the embeddings as numpy array.

        Returns:
            A tuple consisting of a tensor or ndarray of embeddings
            with shape n_images x num_ftrs and labels, fnames

        Examples:
            >>> # embed images in vector space
            >>> embeddings, labels, fnames = encoder.embed(dataloader)

        """

        self.model.eval()
        embeddings, labels, fnames = None, None, []

        if lightly._is_prefetch_generator_available():
            pbar = tqdm(BackgroundGenerator(dataloader, max_prefetch=3),
                        total=len(dataloader))
        else:
            pbar = tqdm(dataloader, total=len(dataloader))

        efficiency = 0.
        embeddings = []
        labels = []
        with torch.no_grad():

            start_time = time.time()
            for (img, label, fname) in pbar:

                img = img.to(device)
                label = label.to(device)

                fnames += [*fname]

                batch_size = img.shape[0]
                prepare_time = time.time()

                emb = self.model.backbone(img)
                emb = emb.detach().reshape(batch_size, -1)

                embeddings.append(emb)
                labels.append(label)

                process_time = time.time()

                efficiency = \
                    (process_time - prepare_time) / (process_time - start_time)
                pbar.set_description(
                    "Compute efficiency: {:.2f}".format(efficiency))
                start_time = time.time()

            embeddings = torch.cat(embeddings, 0)
            labels = torch.cat(labels, 0)
            if to_numpy:
                embeddings = embeddings.cpu().numpy()
                labels = labels.cpu().numpy()

        return embeddings, labels, fnames
    

    def training_step(self, batch, batch_idx):

        # get the two image transformations
        (x0, x1), _, _ = batch
        # forward pass of the transformations
        (z0, p0, q0), (z1, p1, q1) = self(x0, x1)
        # calculate loss for NNCLR
        if self.current_epoch < self.warmup_epochs: # instance discrimination without nearest neighbor sampling if warming up
            z0 = self.nn_replacer(z0.detach(), update=False)
            z1 = self.nn_replacer(z1.detach(), update=True)
        loss = 0.5 * (self.criterion(z0, p1, q0) + self.criterion(z1, p0, q1))
        # log loss and return
        self.log('loss', loss)
        return loss