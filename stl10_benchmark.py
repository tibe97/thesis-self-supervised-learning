# -*- coding: utf-8 -*-
"""
Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)
You can download the ImageNette dataset from here: https://github.com/fastai/imagenette


Code to reproduce the benchmark results:

| Model       | Epochs | Batch Size | Test Accuracy | Peak GPU usage |
|-------------|--------|------------|---------------|----------------|
| MoCo        |  800   | 256        | 0.83          | 4.4 GBytes     |
| SimCLR      |  800   | 256        | 0.85          | 4.4 GBytes     |
| SimSiam     |  800   | 256        | 0.84          | 4.5 GBytes     |
| BarlowTwins |  200   | 256        | 0.80          | 4.5 GBytes     |
| BYOL        |  200   | 256        | 0.85          | 4.6 GBytes     |
| NNCLR       |  200   | 256        | 0.83          | 4.5 GBytes     |
| NNSimSiam   |  800   | 256        | 0.82          | 4.9 GBytes     |
| NNBYOL      |  800   | 256        | 0.85          | 4.6 GBytes     |


| My runs     | Epochs | Batch Size | Test Accuracy | Peak GPU usage |
|-------------|--------|------------|---------------|----------------|
| diff_elevat.|  800   | 256        | 0.82          |                |
| swept_deluge|  800   | 256        | 0.83          |                |


MY Runs: (we keep the optimizer fixed for now)
    - NNN_0:    my_nn_memory_bank_size = 2048
                temperature=0.1
                warmup_epochs=0
                nmb_prototypes=30
                num_negatives=256
                use_sinkhorn = True
                add_swav_loss = False
    - NNN_1:    my_nn_memory_bank_size = 1024
                temperature=0.5
                warmup_epochs=0
                nmb_prototypes=30
                num_negatives=256
                use_sinkhorn = True
                add_swav_loss = True
"""
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
from torchvision.datasets import STL10
from lightly.models.modules import my_nn_memory_bank
from lightly.utils import BenchmarkModule
from lightly.models.modules import NNMemoryBankModule
from lightly.models.mynet import MyNet
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from benchmark_models import MocoModel, BYOLModel, NNCLRModel, NNNModel, SimCLRModel, SimSiamModel, BarlowTwinsModel,NNBYOLModel, NNNModel_Neg, NNNModel_Pos

num_workers = 12
memory_bank_size = 4096

my_nn_memory_bank_size = 2048
temperature=0.5
warmup_epochs=0
nmb_prototypes=30
num_negatives=256
use_sinkhorn = True
add_swav_loss = False

params_dict = dict({
    "memory_bank_size": my_nn_memory_bank_size,
    "temperature": temperature,
    "warmup_epochs": warmup_epochs,
    "nmb_prototypes": nmb_prototypes,
    "num_negatives": num_negatives,
    "use_sinkhorn": use_sinkhorn,
    "add_swav_loss": add_swav_loss
})

logs_root_dir = ('stl10_logs')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 400
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

# The dataset structure should be like this:

path_to_train = 'STL10/unlabeled_images/'
path_to_train_kNN = 'STL10/train_images/'
path_to_test = 'STL10/test_images/'

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

dataset_train_kNN = STL10('STL10/', split="train", download=False, transform=test_transforms)
dataset_test = STL10('STL10/', split="test", download=False, transform=test_transforms)
"""
# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset(
    input_dir=path_to_train,
    transform=test_transforms
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test,
    transform=test_transforms
)
"""

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


"""
model_names = ['MoCo_256', 'SimCLR_256', 'SimSiam_256', 'BarlowTwins_256', 
               'BYOL_256', 'NNCLR_256', 'NNSimSiam_256', 'NNBYOL_256']

models = [MocoModel, SimCLRModel, SimSiamModel, BarlowTwinsModel, 
          BYOLModel, NNCLRModel, NNSimSiamModel, NNBYOLModel]
"""

model_names = ["NNN"]
models = [NNNModel]


bench_results = []
gpu_memory_usage = []

# loop through configurations and train models
for batch_size in batch_sizes:
    for model_name, BenchmarkModel in zip(model_names, models):
        runs = []
        for seed in range(n_runs):
            pl.seed_everything(seed)
            dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
            benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)
            if model_name in ["NNN", "NNN_Pos", "NNN_Neg"]:
                benchmark_model = BenchmarkModel(dataloader_train_kNN, 
                                                classes, warmup_epochs, 
                                                nmb_prototypes, 
                                                my_nn_memory_bank_size, 
                                                use_sinkhorn, 
                                                temperature, 
                                                num_negatives, 
                                                add_swav_loss)

            #logger = TensorBoardLogger('imagenette_runs', version=model_name)
            logger = WandbLogger(project="STL10_knn_validation")  
            logger.log_hyperparams(params=params_dict)

            
            trainer = pl.Trainer(max_epochs=max_epochs, 
                                gpus=gpus,
                                logger=logger,
                                distributed_backend=distributed_backend,
                                default_root_dir=logs_root_dir)
            trainer.fit(
                benchmark_model,
                train_dataloader=dataloader_train_ssl,
                val_dataloaders=dataloader_test
            )
            gpu_memory_usage.append(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            runs.append(benchmark_model.max_accuracy)

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