import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop
import lightly
from lightly.models.modules import my_nn_memory_bank
from lightly.utils import BenchmarkModule
from lightly.models.modules import NNMemoryBankModule
from lightly.models.mynet import MyNet
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

num_workers = 12
memory_bank_size = 4096
# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 400
knn_k = 200
knn_t = 0.1
classes = 10
input_size=128
num_ftrs=512
nn_size=2 ** 16
# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0

class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        #resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        #s#elf.backbone = nn.Sequential(
        #    *list(resnet.children())[:-1],
        #    nn.AdaptiveAvgPool2d(1),
        #)

        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )

        # create a moco model based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(self.backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=True)
        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*self.resnet_moco(x0, x1))
        loss_2 = self.criterion(*self.resnet_moco(x1, x0))
        loss = 0.5 * (loss_1 + loss_2)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=num_ftrs)
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simclr.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class NNCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.NNCLR(self.backbone, num_ftrs=num_ftrs, num_mlp_layers=2)
        self.criterion = lightly.loss.NTXentLoss()

        self.nn_replacer = NNMemoryBankModule(size=nn_size)

    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z0, p0), (z1, p1) = self.resnet_simclr(x0, x1)
        z0 = self.nn_replacer(z0.detach(), update=False)
        z1 = self.nn_replacer(z1.detach(), update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simclr.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class NNNModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, 
                num_classes, 
                warmup_epochs: int=0, 
                nmb_prototypes: int=30, 
                mem_size: int=2048,
                use_sinkhorn: bool=True,
                temperature: float=0.1,
                num_negatives: int=256,
                add_swav_loss: bool=False,
                false_negative_remove: bool=False):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a simclr model based on ResNet
        self.model = \
            MyNet(self.backbone, nmb_prototypes=nmb_prototypes, num_ftrs=num_ftrs, num_mlp_layers=2)
        
        self.nn_replacer = MyNNMemoryBankModule(self.model, size=mem_size, gpus=gpus, use_sinkhorn=use_sinkhorn, false_neg_remove=false_negative_remove)
        self.criterion = MyNTXentLoss(temperature=temperature, num_negatives=num_negatives, add_swav_loss=add_swav_loss)
        self.warmup_epochs = warmup_epochs
        self.num_negatives = num_negatives

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_idx):
        """
        # Trying to place it before passing the inputs through the model
        with torch.no_grad():
            w = self.model.prototypes_layer.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes_layer.weight.copy_(w)
            torch.autograd.set_detect_anomaly(True)
        """
        # get the two image transformations
        (x0, x1), _, _ = batch
        # forward pass of the transformations, plus cluster assignments Q from SwAV module
        (z0, p0, q0), (z1, p1, q1) = self.model(x0, x1)

        if self.current_epoch > self.warmup_epochs-1:
            # sample neighbors, similarities with the sampled negatives and the cluster 
            # assignements of the original Z
            z0, neg0, q0_assign = self.nn_replacer(z0.detach(), self.num_negatives, update=False) 
            z1, neg1, q1_assign = self.nn_replacer(z1.detach(), self.num_negatives, update=True)
           
            loss0, swav_loss0, c_loss0 = self.criterion(z0, p1, q0_assign, q1, neg1) # return swav_loss for the plots
            loss1, swav_loss1, c_loss1 = self.criterion(z1, p0, q1_assign, q0, neg0)
            loss = 0.5 * (loss0 + loss1)
            # loss = 0.5 * (self.criterion(z0, p1, q0_assign, q1, neg1) + self.criterion(z1, p0, q1_assign, q0, neg0))
            #loss = 0.5 * (self.criterion(z0, p1, q0_assign, q1, None) + self.criterion(z1, p0, q1_assign, q0, None))
            if swav_loss1 is not None:
                self.log('train_swav_loss', 0.5*(swav_loss0 + swav_loss1))
            self.log('train_contrastive_loss', 0.5*(c_loss0 + c_loss1))
        else:
            # warming up with classical instance discrimination of same augmented image
            _, _, q0_assign = self.nn_replacer(z0.detach(), self.num_negatives, update=False) 
            _, _, q1_assign = self.nn_replacer(z1.detach(), self.num_negatives, update=False)

            # q tensors are just placeholders, we use them for the SwAV loss only for Swapped Prediction Task
            loss0, swav_loss0, c_loss0 = self.criterion(z0, p1, q0_assign, q1, None) # return swav_loss for the plots
            loss1, swav_loss1, c_loss1 = self.criterion(z1, p0, q1_assign, q0, None)
            loss = 0.5 * (loss0 + loss1)
            self.log('train_swav_loss', 0.5*(swav_loss0 + swav_loss1))
            self.log('train_contrastive_loss', 0.5*(c_loss0 + c_loss1))
        # log loss and return
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class NNNModel_Neg(BenchmarkModule):
    """
    NNN model without positive sampling of nearest neighbors. Just hard negative mining
    """
    def __init__(self, dataloader_kNN, 
                num_classes, 
                warmup_epochs: int=0, 
                nmb_prototypes: int=30, 
                mem_size: int=2048,
                use_sinkhorn: bool=True,
                temperature: float=0.1,
                num_negatives: int=256,
                add_swav_loss: bool=False,
                false_negative_remove: bool=False):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a simclr model based on ResNet
        self.model = \
            MyNet(self.backbone, nmb_prototypes=nmb_prototypes, num_ftrs=num_ftrs, num_mlp_layers=2, out_dim=256)
        
        self.nn_replacer = MyNNMemoryBankModule(self.model, size=mem_size, gpus=gpus, use_sinkhorn=use_sinkhorn, false_neg_remove=false_negative_remove)
        #self.criterion = lightly.loss.NTXentLoss()
        self.criterion = MyNTXentLoss(temperature=temperature, num_negatives=num_negatives, add_swav_loss=add_swav_loss)
        self.warmup_epochs = warmup_epochs
        self.num_negatives = num_negatives

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_idx):
        """
        
        # Trying to place it before passing the inputs through the model
        with torch.no_grad():
            w = self.model.prototypes_layer.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes_layer.weight.copy_(w)
            torch.autograd.set_detect_anomaly(True)
        """

        # get the two image transformations
        (x0, x1), _, _ = batch
        # forward pass of the transformations
        (z0, p0, q0), (z1, p1, q1) = self.model(x0, x1)
        # calculate loss for NNCLR
        if self.current_epoch > self.warmup_epochs-1:
            # sample neighbors, similarities with the sampled negatives and the cluster 
            # assignements of the original Z
            _, neg0, q0_assign = self.nn_replacer(z0.detach(), self.num_negatives, update=False) 
            _, neg1, q1_assign = self.nn_replacer(z1.detach(), self.num_negatives, update=True)

            loss0, swav_loss0, c_loss0 = self.criterion(z0, p1, q0_assign, q1, neg1) # return swav_loss for the plots
            loss1, swav_loss1, c_loss1 = self.criterion(z1, p0, q1_assign, q0, neg0)
            loss = 0.5 * (loss0 + loss1)
            # loss = 0.5 * (self.criterion(z0, p1, q0_assign, q1, neg1) + self.criterion(z1, p0, q1_assign, q0, neg0))
            #loss = 0.5 * (self.criterion(z0, p1, q0_assign, q1, None) + self.criterion(z1, p0, q1_assign, q0, None))
            if swav_loss1 is not None:
                self.log('train_swav_loss', 0.5*(swav_loss0 + swav_loss1))
            self.log('train_contrastive_loss', 0.5*(c_loss0 + c_loss1))
        else:
            # warming up with classical instance discrimination of same augmented image
            _, _, q0_assign = self.nn_replacer(z0.detach(), self.num_negatives, update=False) 
            _, _, q1_assign = self.nn_replacer(z1.detach(), self.num_negatives, update=False)

            # q tensors are just placeholders, we use them for the SwAV loss only for Swapped Prediction Task
            loss0, swav_loss0, c_loss0 = self.criterion(z0, p1, q0_assign, q1, None) # return swav_loss for the plots
            loss1, swav_loss1, c_loss1 = self.criterion(z1, p0, q1_assign, q0, None)
            loss = 0.5 * (loss0 + loss1)
            self.log('train_swav_loss', 0.5*(swav_loss0 + swav_loss1))
            self.log('train_contrastive_loss', 0.5*(c_loss0 + c_loss1))
        # log loss and return
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class NNNModel_Pos(BenchmarkModule):
    """
    Only positive sampling of nearest neighbors, while negatives are used as in NNCLR.
    Different from NNCLR because the neighbors are checked to belong to same cluster.
    """
    def __init__(self, dataloader_kNN, 
                num_classes, 
                warmup_epochs: int=0, 
                nmb_prototypes: int=30, 
                mem_size: int=2048,
                use_sinkhorn: bool=True,
                temperature: float=0.1,
                num_negatives: int=256,
                add_swav_loss: bool=False,
                false_negative_remove: bool=False):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a simclr model based on ResNet
        self.model = \
            MyNet(self.backbone, nmb_prototypes=nmb_prototypes, num_ftrs=num_ftrs, num_mlp_layers=2)
        
        self.nn_replacer = MyNNMemoryBankModule(self.model, size=mem_size, gpus=gpus, use_sinkhorn=use_sinkhorn, false_neg_remove=false_negative_remove)
        #self.criterion = lightly.loss.NTXentLoss()
        self.criterion = self.criterion = MyNTXentLoss(temperature=temperature, num_negatives=num_negatives, add_swav_loss=add_swav_loss)
        self.warmup_epochs = warmup_epochs
        self.num_negatives = num_negatives

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_idx):
        # Trying to place it before passing the inputs through the model
        '''
        with torch.no_grad():
            w = self.model.prototypes_layer.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes_layer.weight.copy_(w)
            torch.autograd.set_detect_anomaly(True)
        '''
        # get the two image transformations
        (x0, x1), _, _ = batch
        # forward pass of the transformations
        (z0, p0, q0), (z1, p1, q1) = self.model(x0, x1)
        # calculate loss for NNCLR
        if self.current_epoch > self.warmup_epochs-1:
            # sample neighbors, similarities with the sampled negatives and the cluster 
            # assignements of the original Z
            z0, _, q0_pred = self.nn_replacer(z0.detach(), self.num_negatives, update=False) 
            z1, _, q1_pred = self.nn_replacer(z1.detach(), self.num_negatives, update=True)
           
            loss = 0.5 * (self.criterion(z0, p1, q0_pred, q1, None) + self.criterion(z1, p0, q1_pred, q0, None))
        else:
            # warming up with classical instance discrimination of same augmented image
            # q tensors are just placeholders, we use them for the SwAV loss only for Swapped Prediction Task
            loss = 0.5 * (self.criterion(z0, p1, q0, q1, None) + self.criterion(z1, p0, q1, q0, None))
        # log loss and return
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=num_ftrs, num_mlp_layers=2)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class NNSimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=num_ftrs, num_mlp_layers=2)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

        self.nn_replacer = NNMemoryBankModule(size=nn_size)

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z0, p0), (z1, p1) = self.resnet_simsiam(x0, x1)
        z0 = self.nn_replacer(z0.detach(), update=False)
        z1 = self.nn_replacer(z1.detach(), update=True)
        loss = self.criterion((z0, p0), (z1, p1))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a barlow twins model based on ResNet
        self.resnet_barlowtwins = \
            lightly.models.BarlowTwins(
                self.backbone, 
                num_ftrs=num_ftrs,
                proj_hidden_dim=2048,
                out_dim=2048,
                num_mlp_layers=2
                )
        self.criterion = lightly.loss.BarlowTwinsLoss()

    def forward(self, x):
        self.resnet_barlowtwins(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_barlowtwins(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_barlowtwins.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a byol model based on ResNet
        self.resnet_byol = \
            lightly.models.BYOL(self.backbone, num_ftrs=num_ftrs)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        self.resnet_byol(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_byol(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_byol.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class NNBYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        # create a byol model based on ResNet
        self.resnet_byol = \
            lightly.models.BYOL(self.backbone, num_ftrs=num_ftrs)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

        self.nn_replacer = NNMemoryBankModule(size=nn_size)

    def forward(self, x):
        self.resnet_byol(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        (z0, p0), (z1, p1) = self.resnet_byol(x0, x1)
        z0 = self.nn_replacer(z0.detach(), update=False)
        z1 = self.nn_replacer(z1.detach(), update=True)
        loss = self.criterion((z0, p0), (z1, p1))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_byol.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]