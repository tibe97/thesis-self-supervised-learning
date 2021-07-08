from logging import critical
import torch
import torchvision
import torch.nn as nn
import argparse
import pytorch_lightning as pl
import lightly.models as models
import lightly.loss as loss
import lightly.data as data
import lightly.embedding as embedding

from lightly.embedding.embedding import NNCLRSelfSupervisedEmbedding
from lightly.models.mynet import MyNet
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule

def main(args):
    gpus = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if gpus==1 else 'cpu'
    max_epochs=args.max_epochs
    batch_size = args.batch_size
    input_size = args.input_size
    memory_bank_size = args.memory_bank_size
    num_clusters = args.num_clusters
    use_sinkhorn = args.use_sinkhorn

    training_info = "Starting training with: \nmax_epochs: {} \nbatch_size: {} \ninput_size: {} \nmemory_bank_size: {} \nnum_clusters: {} \nuse_sinkhorn: {}\n".format(
        max_epochs,
        batch_size,
        input_size,
        memory_bank_size,
        num_clusters,
        use_sinkhorn
    )
    print(training_info)


    # the collate function applies random transforms to the input images
    collate_fn = data.ImageCollateFunction(input_size=input_size, cj_prob=0.5)

    # create a dataset from your image folder
    dataset = data.LightlyDataset(input_dir='imagenette2-160/train')

    # build a PyTorch dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,                # pass the dataset to the dataloader
        batch_size,             # a large batch size helps with the learning
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
    model = MyNet(backbone=backbone, nmb_prototypes=num_clusters)
    nn_replacer = MyNNMemoryBankModule(model, size=memory_bank_size, gpus=gpus, use_sinkhorn=use_sinkhorn)

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

"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Self-Supervised Learning')

    # Flatland env parameters
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--input-size', type=int, default=32,
                        help='Image input size')
    parser.add_argument('--memory-bank-size', type=int, default=2**16,
                        help='Size of the memory bank')
    parser.add_argument('---num-clusters', type=int, default=3000,
                        help='Number of clusters. Default for full Imagenet is 3000')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed')
    parser.add_argument('--use-sinkhorn', type=bool, default=False,
                        help='Whether to use Sinkhorn algorithm when assigning clusters')

"""