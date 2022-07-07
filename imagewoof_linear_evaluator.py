import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
import copy
from argparse import ArgumentParser
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.transforms import CenterCrop
from lightly.models.modules import my_nn_memory_bank
from lightly.models.nnclr import NNCLR
from lightly.utils import BenchmarkModule
from lightly.models.modules import NNMemoryBankModule
from lightly.models.mynet import MyNet
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss


from lightly.linear_evaluation.ssl_finetuner import SSLFineTuner
from lightly.linear_evaluation.transforms import SwAVFinetuneTransform
from lightly.linear_evaluation.dataset_normalizations import imagenet_normalization, stl10_normalization
from lightly.linear_evaluation.imagenet_datamodule import ImagenetDataModule
from benchmark_models import MocoModel, BYOLModel, NNCLRModel, NNNModel, NegMining_TrueLabels, SimCLRModel, SimSiamModel, BarlowTwinsModel,NNBYOLModel, NNNModel_Neg, NNNModel_Pos, SupervisedClustering, SwAVModel, PosMining_TrueLabels, PosMining_FalseNegRemove_TrueLabels, NNNModel_Neg_Momentum, Moco_NNCLR_Model


num_workers = 12
memory_bank_size = 4096

my_nn_memory_bank_size = 1024
temperature=0.1
warmup_epochs=0
nmb_prototypes=30
num_negatives=256
use_sinkhorn = True
add_swav_loss = True

params_dict = dict({
    "memory_bank_size": my_nn_memory_bank_size,
    "temperature": temperature,
    "warmup_epochs": warmup_epochs,
    "nmb_prototypes": nmb_prototypes,
    "num_negatives": num_negatives,
    "use_sinkhorn": use_sinkhorn,
    "add_swav_loss": add_swav_loss
})

logs_root_dir = ('imagenette_logs')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 800
knn_k = 200
knn_t = 0.1
classes = 120
input_size=128
num_ftrs=512
nn_size=2 ** 16

# benchmark
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_sizes = [256]

# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0
distributed_backend = 'ddp' if torch.cuda.device_count() > 1 else None

# The dataset structure should be like this:

path_to_dir = 'data/Images'

dm = ImagenetDataModule(data_dir=path_to_dir, batch_size=batch_sizes[0], num_workers=num_workers)



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

img_dataset = ImageFolder(path_to_dir)

total_count = len(img_dataset)
train_count = int(0.8 * total_count)
valid_count = total_count - train_count


train_dataset, valid_dataset = torch.utils.data.random_split(
    img_dataset, (train_count, valid_count)
)

ssl_train_dataset = copy.deepcopy(train_dataset.dataset)
dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(ssl_train_dataset)
# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(copy.deepcopy(train_dataset.dataset), transform=test_transforms)
dataset_train_kNN.test_mode = True

dataset_test = lightly.data.LightlyDataset.from_torch_dataset(copy.deepcopy(valid_dataset.dataset), transform=test_transforms)
dataset_test.test_mode = True

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



def cli_main():  # pragma: no cover

    seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='stl10, imagenet', default='stl10')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='checkpoints/ImageWoof120/pos_mining_epoch=380.ckpt')
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


    model_names = ["NNCLR"]
    models = [NNNModel_Pos]

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
                benchmark_model = BenchmarkModel.load_from_checkpoint(checkpoint_path=ckpt_path, dataloader_kNN=dataloader_train_kNN, num_classes=30)


                logger = WandbLogger(project="ssl_linear_evaluation_imagewoof120")  
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