import numpy as np
import argparse
import pandas as pd
import shutil, time, os, requests, random, copy, collections, sys, datetime
from itertools import permutations 
import seaborn as sns
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as tF
import torch.optim as optim
from torchvision import datasets, transforms, models

# from torchvision.models.utils import load_state_dict_from_url

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#%matplotlib inline

from dataloaders import MRNetDataModule
from transformations import *
from model_utils import ClassificationModel
import losses
import byol3d
from trainer import Trainer
from utils import run_command

import copy
import os

import time
import lightly
import multiprocessing as mp
import pytorch_lightning as pl
import torchvision
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models import utils
from lightly.utils import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger

logs_root_dir = os.path.join(os.getcwd(), 'benchmark_logs')
# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True). 
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating 
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all 
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False 

np.random.seed(1234)
torch.manual_seed(1234)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus',type=int,default=1)
    parser.add_argument('--pretrain',type=bool,default=True,help = 'Stage of learning')
    parser.add_argument('--model_name',type=str,default='simclr',help = 'Name of Algorithm or Framework')
    parser.add_argument('--base_encoder_name',type=str,default='resnet50')
    parser.add_argument('--projector_type',type=str,default='nonlinear')
    parser.add_argument('--proj_num_layers',type=int,default=2)
    parser.add_argument('--projector_hid_dim',type=int,default=2048)
    parser.add_argument('--projector_out_dim',type=int,default=128)
    parser.add_argument('--proj_use_bn', action='store_false')
    parser.add_argument('--proj_last_bn', action='store_true')
    parser.add_argument('--predictor_type',type=str,default='nonlinear')
    parser.add_argument('--pred_num_layers',type=int,default=None)
    parser.add_argument('--predictor_hid_dim',type=int,default=None)
    parser.add_argument('--predictor_out_dim',type=int,default=None)
    parser.add_argument('--pred_use_bn',action='store_false')
    parser.add_argument('--pred_last_bn',action='store_true')
    parser.add_argument('--K',type=int,default=65536)
    parser.add_argument('--m',type=float,default=0.99)
    parser.add_argument('--optimizer',type=str,default='sgd')
    parser.add_argument('--lr',type=float,default=0.3)
    parser.add_argument('--lr_scheduler',type=str,default='lwca')
    parser.add_argument('--momentum',type=float, default = 0.9)
    parser.add_argument('--weight_decay',type=float,default=1e-6)
    parser.add_argument('--temperature',type=float,default=0.5)
    parser.add_argument('--color_distortion_strength',type=float,default=0.5)
    parser.add_argument('--train_epochs',type=int,default=100)
    parser.add_argument('--warmup_epochs',type=int,default=10)
    parser.add_argument('--max_epochs',type=int,default=1000)
    parser.add_argument('--warmup_start_lr',type=float,default=1e-4)
    parser.add_argument('--eta_min',type=float,default=1e-4)
    parser.add_argument('--pretrain_batch_size',type=int,default=64)
    parser.add_argument('--ds_batch_size',type=int,default=1)
    parser.add_argument('--lineval_epochs',type=int,default=20)
    parser.add_argument('--lineval_optim',type=str,default='sgd')
    parser.add_argument('--lineval_lr',type=float,default=0.05)
    parser.add_argument('--lineval_momentum',type=float,default=0.9)
    parser.add_argument('--lineval_wd',type=float,default=0.0)
    parser.add_argument('--lineval_lr_schedule',type=str,default='multistep')
    parser.add_argument('--finetune_epochs',type=int,default=30)
    parser.add_argument('--finetune_optim',type=str,default='sgd')
    parser.add_argument('--finetune_lr',type=float,default=0.01)
    parser.add_argument('--finetune_momentum',type=float,default=0.9)
    parser.add_argument('--finetune_wd',type=float,default=0.0)
    parser.add_argument('--finetune_lr_schedule',type=str,default='multistep')
    parser.add_argument('--knn_fracs', type=float, default=1.0)
    parser.add_argument('--num_neighbours', type=int, default=20)
    parser.add_argument('--knn_wt_type', type=str, default='distance')
    parser.add_argument('--knn_algo_type', type=str, default='auto')
    parser.add_argument('--knn_metric_type', type=str, default='minkowski')

    parser.add_argument('--data_dims',type=str,default='32x32')
    parser.add_argument('--dataset',type=str,default='cifar10',help = 'Name of Dataset')
    parser.add_argument('--dataset_path',type=str,default=os.path.join(os.getcwd(),'datasets'))
    parser.add_argument('--download',action='store_true') #argparse.BooleanOptionalAction in Python 3.9
    parser.add_argument('--modelsavepath',type=str,default=os.path.join(os.getcwd(),'saved_models'))
    parser.add_argument('--modelsaveinterval',type=int,default=10)
    parser.add_argument('--resume',action='store_true') #argparse.BooleanOptionalAction in Python
    parser.add_argument('--model_path',type=str, default=os.path.join(os.getcwd(),'saved_models'))
    #extra args
    parser.add_argument('--bn_bias_lr',type=float, default = None)
    parser.add_argument('--ft_fc_lr',type=float, default = None)
    parser.add_argument('--lambd',type=float, default = 1.0)
    parser.add_argument('--mu',type=float, default = 1.0)
    parser.add_argument('--nu',type=float, default = 1.0)

    parser.add_argument('--n_temperature', type=float, default = None)
    parser.add_argument('--lambda_loss', type=float, default = 1.0)

    parser.add_argument('--step_size', type=int, default = 10)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_milestones', type=str, default=None)
    parser.add_argument('--lr_factor', type=float, default=None)
    parser.add_argument('--lr_total_iters', type=int, default=None)

    parser.add_argument('--run_num', type=str, default=None)
    parser.add_argument('--grad_acc', action = 'store_true')
    parser.add_argument('--acc_bs', type = int, default = 1)

    parser.add_argument('--pt_num_frames', type=int, default=16)
    parser.add_argument('--ds_num_frames', type=int, default=16)

    parser.add_argument('--remarks', type=str, default=None)
    parser.add_argument('--lowest_val_eval', action='store_true')

    args = parser.parse_args()

    #SYSTEM INFORMATION
    command = ['nvidia-smi']
    run_command(command)
    #more system INFORMATION
    command = ['python', '-m', 'torch.utils.collect_env']
    run_command(command)

    # =============================================================================

    date = datetime.datetime.now().strftime("%d-%m-%Y")
    dict_args = vars(args)
    args_table = [[k,v] for k,v in dict_args.items()]
    print(tabulate(args_table,headers = ["Argument","Value"], tablefmt = "pretty"))

    #print(args.download, type(args.download), 'True' if args.download == True else 'False')

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.modelsavepath):
        os.makedirs(args.modelsavepath)

    # =============================================================================

	# mrnet_path = 'E:/Siladittya_JRF/Dataset/MRNet-v1.0'

	# =============================================================================
    benchmark_model = byol3d.BYOLModel(**dict_args)
    benchmark_model.acc_iters = args.pretrain_batch_size//args.acc_bs
    transforms_ = BYOLTransform(int(args.data_dims.split('x')[0]))

    if args.dataset == 'mrnet':
    	class_name = 'acl' #input("Enter class name : [acl / abn / men] : ")
    	plane = 'sagittal' #input("Enter class name : [sagittal / coronal / axial] : ")
    	dm = MRNetDataModule(args.dataset_path,
    						 class_name,
    						 plane,
                             args.pt_num_frames,
                             args.ds_num_frames,
    						 args.acc_bs,
                             args.ds_batch_size,
    						 transforms_,
                             224,
                             dsoversample = True,
                             dsbinary = True)
    elif args.dataset == 'rsnabrats':
        class_name = 'MGMT_value'
        plane = 'T1w'
        dm = RSNABraTSModule(args.dataset_path,
                             class_name,
                             plane,
                             args.pt_num_frames,
                             args.ds_num_frames,
                             args.acc_bs,
                             args.ds_batch_size,
                             transforms_,
                             224,
                             dsoversample = True,
                             dsbinary = True)

    # =============================================================================
    runs = []
    dm.setup(stage = 'train',pretrain = True)
    dm.setup(stage = 'valid',pretrain = True)
    train_loader = dm.train_dataloader(True)
    valid_loader = dm.valid_dataloader(True)
    # RELOAD MODELS IF RESUME TRUE
    if args.resume:
        state = torch.load(args.model_path)
        model.load_state_dict(state['model_state_dict'])

    # =============================================================================
    model_name = 'byol3d'
    bench_results = dict()
    experiment_version = None

    sub_dir = model_name if n_runs <= 1 else f'{model_name}/run{seed}'
    logger = TensorBoardLogger(
                save_dir=os.path.join(logs_root_dir, args.dataset),
                name='',
                sub_dir=sub_dir,
                version=experiment_version,
            )
    if experiment_version is None:
        # Save results of all models under same version directory
        experiment_version = logger.version
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        every_n_epochs = 50,
        save_top_k = -1
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        gpus= args.gpus,
        default_root_dir=logs_root_dir,
        strategy=distributed_backend,
        sync_batchnorm=sync_batchnorm,
        logger=logger,
        accumulate_grad_batches = args.pretrain_batch_size//args.acc_bs,
        callbacks=[checkpoint_callback])#, GradCallback()]
    #)
    start = time.time()
    trainer.fit(
        benchmark_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )
    end = time.time()
    run = {
        'model': model_name,
        'batch_size': args.batch_size,
        'epochs': args.max_epochs,
        'runtime': end - start,
        'gpu_memory_usage': torch.cuda.max_memory_allocated(),
        'seed': seed,
    }
    runs.append(run)
    print(run)

    # delete model and trainer + free up cuda memory
    del benchmark_model
    del trainer
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    bench_results[model_name] = runs

    # print results table
    header = (
        f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
        f"| {'Time':>10} | {'Peak GPU Usage':>14} |"
    )
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for model, results in bench_results.items():
        runtime = np.array([result['runtime'] for result in results])
        runtime = runtime.mean() / 60 # convert to min
        # accuracy = np.array([result['max_accuracy'] for result in results])
        gpu_memory_usage = np.array([result['gpu_memory_usage'] for result in results])
        gpu_memory_usage = gpu_memory_usage.max() / (1024**3) # convert to gbyte

        # if len(accuracy) > 1:
            # accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
        # else:
            # accuracy_msg = f"{accuracy.mean():>18.3f}"

        print(
            f"| {model:<13} | {args.batch_size:>10} | {args. max_epochs:>6} "
            f"| {runtime:>6.1f} Min "
            f"| {gpu_memory_usage:>8.1f} GByte |",
            flush=True
        )
    print('-' * len(header))

    # =============================================================================
    #BUILD A MODEL FOR THE DOWNSTREAM TASK
    print("=============================")
    print("|  STARTING DOWNSTREAM TASK |")
    print("=============================")
    ds_model = ClassificationModel(args.base_encoder_name,
                                   dm.num_classes, 
                                   args.data_dims,
                                   classification_type = 'binary').to('cuda:0')

    dm.setup(stage = 'train',pretrain = False)
    dm.setup(stage = 'valid',pretrain = False)
    train_loader = dm.train_dataloader(False)
    valid_loader = dm.valid_dataloader(False)
    runs = []
    bench_results = dict()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        every_n_epochs = 50,
        save_top_k = -1
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        gpus= args.gpus,
        default_root_dir=logs_root_dir,
        strategy=distributed_backend,
        sync_batchnorm=sync_batchnorm,
        logger=logger,
        accumulate_grad_batches = args.ds_batch_size//args.acc_bs,
        callbacks=[checkpoint_callback])#, GradCallback()]
    #)
    start = time.time()
    trainer.fit(
        ds_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )
    end = time.time()
    run = {
        'model': model_name,
        'batch_size': args.batch_size,
        'epochs': args.max_epochs,
        'runtime': end - start,
        'gpu_memory_usage': torch.cuda.max_memory_allocated(),
        'seed': seed,
    }
    runs.append(run)
    print(run)

    # delete model and trainer + free up cuda memory
    # del benchmark_model
    # del trainer
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    bench_results[model_name] = runs

    # print results table
    header = (
        f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
        f"| {'Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
    )
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for model, results in bench_results.items():
        runtime = np.array([result['runtime'] for result in results])
        runtime = runtime.mean() / 60 # convert to min
        # accuracy = np.array([result['max_accuracy'] for result in results])
        gpu_memory_usage = np.array([result['gpu_memory_usage'] for result in results])
        gpu_memory_usage = gpu_memory_usage.max() / (1024**3) # convert to gbyte

        if len(accuracy) > 1:
            accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
        else:
            accuracy_msg = f"{accuracy.mean():>18.3f}"

        print(
            f"| {model:<13} | {args.batch_size:>10} | {args. max_epochs:>6} "
            f"| {accuracy_msg} | {runtime:>6.1f} Min "
            f"| {gpu_memory_usage:>8.1f} GByte |",
            flush=True
        )
    print('-' * len(header))

    trainer.test(dataloaders=valid_loader)

    # =============================================================================
    #KNN EVALUATION
    # knn_eval_metrics = trainer.knn_eval(ds_model, 
    #                                     fracs = args.knn_fracs, 
    #                                     k = args.num_neighbours, 
    #                                     weights = args.knn_wt_type, 
    #                                     algorithm = args.knn_algo_type, 
    #                                     metric = args.knn_metric_type)

    # =============================================================================
    # LINEAR EVALUATION
    #ds_model = ClassificationModel('resnet18',dm.num_classes).to('cuda:0')
    # lin_eval_metrics = trainer.linear_eval(ds_model)

    # =============================================================================
    #FINE TUNING
    # fine_tune_metrics = trainer.fine_tune(ds_model)

    # # =============================================================================

    # metrics_dict = {**fine_tune_metrics} # {**knn_eval_metrics, **lin_eval_metrics, 
    # # print(metrics_dict)
    # # print(dict_args)

    # # =============================================================================

    # trainer.writer.add_hparams(dict_args, metrics_dict,
    #                             run_name = '_'.join([args.model_name,
    #                                                 args.base_encoder_name,
    #                                                 args.optimizer,'lr',str(args.lr),
    #                                                 'bs',str(args.pretrain_batch_size),
    #                                                 'ep',str(args.train_epochs), date]))

    # # =============================================================================

    # trainer.writer.close()

    # =============================================================================