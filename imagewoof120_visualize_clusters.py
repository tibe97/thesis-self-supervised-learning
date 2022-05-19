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
from torchvision.datasets import ImageFolder
import lightly
from lightly.models.modules import my_nn_memory_bank
from lightly.utils import BenchmarkModule
from lightly.models.modules import NNMemoryBankModule
from lightly.models.mynet import MyNet
from lightly.models.modules.my_nn_memory_bank import MyNNMemoryBankModule
from lightly.loss.my_ntx_ent_loss import MyNTXentLoss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from benchmark_models import MocoModel, BYOLModel, NNCLRModel, NNNModel, SimCLRModel, SimSiamModel, BarlowTwinsModel,NNBYOLModel, NNNModel_Neg, NNNModel_Pos
#import tensorflow as tf
#from tensorboard.plugins import projector

checkpoint_path = "checkpoints/ImageWoof120/ImageWoof_NEG_SoftNeg.ckpt"
num_workers = 2
memory_bank_size = 4096

my_nn_memory_bank_size = 4096
temperature=0.5
warmup_epochs=0
nmb_prototypes=300
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

logs_dir = ('tb_logs/imagewoof120')
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
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_sizes = [512]

# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0
distributed_backend = 'ddp' if torch.cuda.device_count() > 1 else None

# The dataset structure should be like this:

path_to_dir = 'data/Images'


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


#ipdb.set_trace()

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
        shuffle=True,
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
            logger = WandbLogger(project="ssl_imagewoof120_visualize_debug")  
            logger.log_hyperparams(params=params_dict)

            proto_to_class = torch.zeros(nmb_prototypes, classes)
            class_to_protos = torch.zeros(classes, nmb_prototypes)
            proto_to_class_list = []
            class_images = {}
            for step in range(10):
                x, y, _ = next(iter(dataloader_test))
                
                if len(class_images.keys()) < classes:
                    for i,label in enumerate(y):
                        if label.item() not in class_images.keys():
                            class_images[label.item()] = wandb.Image(x[i])
                embeddings, _, _ = benchmark_model.model(x)
                prototypes = benchmark_model.model.prototypes_layer.weight
                batch_similarities, batch_clusters  = benchmark_model.nn_replacer.compute_assignments_batch(embeddings)
                
                

                # For each example in the batch, take its assigned cluster and look-up its real class given by label y
                # After each batch, take argmax of the counting matrix. For each prototype there will be one dominant class.
                # Plot over multiple batches to see if the dominant class for each prototype changes
                for i, cluster in enumerate(batch_clusters):
                    proto_to_class[cluster, y[i]] += 1
                wandb.log({
                    "embeddings": wandb.Table(
                        columns = list(range(prototypes.shape[1])),
                        data = prototypes.tolist()
                    )
                })

                data = [[y, x] for (y, x) in zip(batch_similarities.indices, batch_similarities.values)]
                table = wandb.Table(data=data, columns = ["cluster", "similarity"])
                wandb.log({"batch similarities" : wandb.plot.scatter(table, "cluster", "similarity",
                                                title="Similarities of batch vs clusters")})

                #wandb.log({'prototypes to matrices': wandb.plots.HeatMap(list(range(classes)), list(range(nmb_prototypes)), proto_to_class, show_text=False)})
                
                """
                proto_to_class_table = wandb.Table(data=[[i, torch.argmax(proto_to_class[i,:])] for i in range(nmb_prototypes)], columns = ["prototype", "class"])
                wandb.log({"Prototypes to Class Assignments" : wandb.plot.scatter(proto_to_class_table, "prototype", "class",
                                                title="Class assignment for each prototype")})
                """
                '''
                for i in range(nmb_prototypes):
                    proto_to_class_list.append([i, torch.argmax(proto_to_class[i,:])])
                '''
            '''
            proto_to_class_table = wandb.Table(data=proto_to_class_list, columns = ["prototype", "class"])
            wandb.log({"Prototypes to Class Assignments" : wandb.plot.scatter(proto_to_class_table, "prototype", "class",
                                            title="Class assignment for each prototype")})
            '''
            table_list = []
            for i in range(nmb_prototypes):
                top_3_indices = torch.topk(proto_to_class[i,:], 3)[1]
                top_3_percentages = [proto_to_class[i,k]/torch.sum(proto_to_class[i,:]) for k in top_3_indices]
                table_list.append([i, top_3_indices[0], top_3_percentages[0]*100, class_images[top_3_indices[0].item()], top_3_indices[1], top_3_percentages[1]*100, class_images[top_3_indices[1].item()], top_3_indices[2], top_3_percentages[2]*100, class_images[top_3_indices[2].item()]])
            table_columns = ["prototype", "top1_class", "top1_%", "top1_image", "top2_class", "top2_%", "top2_image", "top3_class", "top3_%", "top3_image"]
            table = wandb.Table(data=table_list, columns=table_columns)
            wandb.log({"Prototypes_table": table})

            # delete model and trainer + free up cuda memory
            del benchmark_model
            torch.cuda.empty_cache()
        bench_results.append(runs)

for result, model, gpu_usage in zip(bench_results, model_names, gpu_memory_usage):
    result_np = np.array(result)
    mean = result_np.mean()
    std = result_np.std()
    print(f'{model}: {mean:.3f} +- {std:.3f}, GPU used: {gpu_usage / (1024.0**3):.1f} GByte', flush=True)