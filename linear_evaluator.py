import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
from argparse import ArgumentParser
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop
from lightly.models.modules import my_nn_memory_bank
from lightly.utils import BenchmarkModule
from lightly.models.modules import NNMemoryBankModule
from lightly.models.mynet import MyNet
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss


from lightly.linear_evaluation.ssl_finetuner import SSLFineTuner
from lightly.linear_evaluation.transforms import SwAVFinetuneTransform
from lightly.linear_evaluation.dataset_normalizations import imagenet_normalization, stl10_normalization
from lightly.linear_evaluation.imagenet_datamodule import ImagenetDataModule



num_workers = 12
memory_bank_size = 4096



logs_root_dir = ('imagenette_logs')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 800
knn_k = 200
knn_t = 0.1
classes = 10
input_size=128
num_ftrs=512
nn_size=2 ** 16

# benchmark
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_sizes = [256]

# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0
distributed_backend = 'ddp' if torch.cuda.device_count() > 1 else None

dataset = 'imagewoof'
# The dataset structure should be like this:
if dataset == 'imagenette':
    data_dir = 'imagenette2-160'
    path_to_train = 'imagenette2-160/train/'
    path_to_test = 'imagenette2-160/val/'
elif dataset == 'imagewoof':
    data_dir = 'imagewoof2-160'
    path_to_train = 'imagewoof2-160/train/'
    path_to_test = 'imagewoof2-160/val/'    

dm = ImagenetDataModule(data_dir=data_dir, batch_size=batch_sizes[0], num_workers=num_workers)

dm.train_transforms = SwAVFinetuneTransform(
    normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=False
)
dm.val_transforms = SwAVFinetuneTransform(
    normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=True
)

dm.test_transforms = SwAVFinetuneTransform(
    normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=True
)


# Use SimCLR augmentations
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(128),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_ssl = lightly.data.LightlyDataset(
    input_dir=path_to_train
)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset(
    input_dir=path_to_train,
    transform=test_transforms
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test,
    transform=test_transforms
)

def get_data_loaders(batch_size: int):
    """Helper method to create dataloaders for ssl, kNN train and kNN test
    Args:
        batch_size: Desired batch size for all dataloaders
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test



class MockupModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
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




def cli_main():  # pragma: no cover

    seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='imagewoof, imagenette', default='imagenette')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='lightly/linear_evaluation/epoch=799-step=28799.ckpt')
    parser.add_argument('--data_dir', type=str, help='path to dataset', default=os.getcwd())

    parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=1, type=int, help="number of GPUs")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument('--in_features', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--learning_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--nesterov', type=bool, default=False)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--final_lr', type=float, default=0.)

    args = parser.parse_args()


    model_names = ["NNN"]
    models = [MockupModel]

    ckpt_path = args.ckpt_path

    bench_results = []
    gpu_memory_usage = []

    # loop through configurations and train models
    for batch_size in batch_sizes:
        for model_name, BenchmarkModel in zip(model_names, models):
            runs = []
            for seed in range(n_runs):
                pl.seed_everything(seed)
                dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
                #benchmark_model = BenchmarkModel(dataloader_train_kNN, dm.num_classes).load_from_checkpoint(ckpt_path, dataloader_train_kNN, dm.num_classes, strict=False)
                benchmark_model = BenchmarkModel().load_from_checkpoint(ckpt_path, strict=False)

                params_dict = dict({
                    "ckpt_path": args.ckpt_path,
                    "dataset": dataset
                })

                project="ssl_linear_evaluation"
                if dataset == 'imagewoof':
                    project += "_imagewoof10"
                logger = WandbLogger(project=project)  
                logger.log_hyperparams(params=params_dict)

                tuner = SSLFineTuner(
                    benchmark_model.backbone,
                    in_features=args.in_features,
                    num_classes=dm.num_classes,
                    epochs=args.num_epochs,
                    hidden_dim=None,
                    dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                    scheduler_type=args.scheduler_type,
                    gamma=args.gamma,
                    final_lr=args.final_lr
                )

                trainer = Trainer(
                    gpus=gpus,
                    num_nodes=1,
                    precision=16,
                    max_epochs=args.num_epochs,
                    distributed_backend=distributed_backend,
                    sync_batchnorm=True if gpus > 1 else False,
                    logger=logger
                )

                trainer.fit(tuner, train_dataloader=dataloader_train_kNN, val_dataloaders=dataloader_test)
                trainer.test(test_dataloaders=dataloader_test)


                # delete model and trainer + free up cuda memory
                del benchmark_model
                del trainer
                torch.cuda.empty_cache()
            bench_results.append(runs)

    for result, model, gpu_usage in zip(bench_results, model_names, gpu_memory_usage):
        result_np = np.array(result)
        mean = result_np.mean()
        std = result_np.std()
        print(f'{model}: {mean:.3f} +- {std:.3f}, GPU used: {gpu_usage / (1024.0**3):.1f} GByte', flush=True)


if __name__ == '__main__':
    cli_main()