# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import copy
import ipdb
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop
from torchvision.datasets import ImageFolder, STL10
import lightly
from lightly.models.modules import my_nn_memory_bank
from lightly.utils import BenchmarkModule
from lightly.models.modules import NNMemoryBankModule
from lightly.models.mynet import MyNet
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from benchmark_models import MocoModel, BYOLModel, NNCLRModel, NNNModel, SimCLRModel, SimSiamModel, BarlowTwinsModel,NNBYOLModel, NNNModel_Neg, NNNModel_Pos, FalseNegRemove_TrueLabels
import tensorflow as tf
from tensorboard.plugins import projector

checkpoint_path = "checkpoint/STL10/NoSwavLoss.ckpt"
num_workers = 2
memory_bank_size = 4096

my_nn_memory_bank_size = 2048
temperature=0.5
warmup_epochs=0
nmb_prototypes=100
num_negatives=512
use_sinkhorn = True
add_swav_loss = False
false_negative_remove = True
soft_neg = False


params_dict = dict({
    "memory_bank_size": my_nn_memory_bank_size,
    "temperature": temperature,
    "warmup_epochs": warmup_epochs,
    "nmb_prototypes": nmb_prototypes,
    "num_negatives": num_negatives,
    "use_sinkhorn": use_sinkhorn,
    "add_swav_loss": add_swav_loss,
    "false_negative_remove": false_negative_remove,
    "soft_neg": soft_neg
})

logs_dir = ('tb_logs/stl10')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 800
knn_k = 200
knn_t = 0.1
classes = 120
input_size=128
num_ftrs=512
nn_size=2 ** 16

# benchmark
n_runs = 10 # optional, increase to create multiple runs and report mean + std
batch_sizes = [512]

# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0
distributed_backend = 'ddp' if torch.cuda.device_count() > 1 else None

# The dataset structure should be like this:

path_to_train = 'data/STL10/unlabeled_images/'
path_to_train_kNN = 'data/STL10/train_images/'
path_to_test = 'data/STL10/test_images/'

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
dataset_train_kNN = lightly.data.LightlyDataset(
    input_dir=path_to_train_kNN,
    transform=test_transforms
)
dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test,
    transform=test_transforms
)
"""
dataset_train_kNN = STL10('STL10/', split="train", download=False, transform=test_transforms)
dataset_test = STL10('STL10/', split="test", download=False, transform=test_transforms)
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
        num_workers=num_workers,
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
model_names = ["NNN_Neg"]
models = [NNNModel_Neg]


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
            if model_name in ["NNN", "NNN_Pos", "NNN_Neg", "FalseNegRemove"]:
                benchmark_model = BenchmarkModel.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                    dataloader_kNN=dataloader_train_kNN, 
                    num_classes=classes,
                    nmb_prototypes=nmb_prototypes, use_sinkhorn=use_sinkhorn)
            benchmark_model.model.eval()
            #logger = TensorBoardLogger('imagenette_runs', version=model_name)
            logger = WandbLogger(project="ssl_stl10_visualize_debug")  
            logger.log_hyperparams(params=params_dict)

            x, y, _ = next(iter(dataloader_test))
           
            embeddings, _, _ = benchmark_model.model(x)
            
            wandb.log({
                "embeddings": wandb.Table(
                    columns = list(range(embeddings.shape[1])),
                    data = embeddings.tolist()
                )
            })
            
            prototypes = benchmark_model.model.prototypes_layer.weight
            batch_similarities  = benchmark_model.nn_replacer.compute_assignments_batch(embeddings)
           
            wandb.log({
                "prototypes": wandb.Table(
                    columns = list(range(prototypes.shape[1])),
                    data = prototypes.tolist()
                )
            })

            data = [[y, x] for (y, x) in zip(batch_similarities.indices, batch_similarities.values)]
            table = wandb.Table(data=data, columns = ["cluster", "similarity"])
            wandb.log({"batch similarities" : wandb.plot.scatter(table, "cluster", "similarity",
                                            title="Similarities of batch vs clusters")})


            
            

            

            # delete model and trainer + free up cuda memory
            del benchmark_model
            torch.cuda.empty_cache()
        bench_results.append(runs)

for result, model, gpu_usage in zip(bench_results, model_names, gpu_memory_usage):
    result_np = np.array(result)
    mean = result_np.mean()
    std = result_np.std()
    print(f'{model}: {mean:.3f} +- {std:.3f}, GPU used: {gpu_usage / (1024.0**3):.1f} GByte', flush=True)