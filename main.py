from logging import critical
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import lightly.models as models
import lightly.loss as loss
import lightly.data as data
import lightly.embedding as embedding

from lightly.embedding.embedding import NNCLRSelfSupervisedEmbedding
from lightly.models.mynet import MyNet
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule


gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus==1 else 'cpu'
max_epochs=10

# the collate function applies random transforms to the input images
collate_fn = data.ImageCollateFunction(input_size=32, cj_prob=0.5)

# create a dataset from your image folder
dataset = data.LightlyDataset(input_dir='imagenette2-160/train')

# build a PyTorch dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,                # pass the dataset to the dataloader
    batch_size=32,         # a large batch size helps with the learning
    shuffle=True,           # shuffling is important!
    collate_fn=collate_fn)  # apply transformations to the input images


resnet = torchvision.models.resnet18()
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, 512, 1),
)

# create the SimCLR model using the newly created backbone
#model = models.SimCLR(backbone, num_ftrs=512)
#model = models.NNCLR(backbone)
model = MyNet(backbone=backbone)
nn_replacer = MyNNMemoryBankModule(model, size=2 ** 16, gpus=gpus)

#criterion = loss.NTXentLoss()
criterion = MyNTXentLoss(nn_replacer)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


"""
encoder = embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader
)
"""

encoder = NNCLRSelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader,
    nn_replacer
)


encoder = encoder.to(device)

encoder.train_embedding(gpus=gpus,
                        progress_bar_refresh_rate=100,
                        max_epochs=max_epochs)