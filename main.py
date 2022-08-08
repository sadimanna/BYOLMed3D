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
    model = byol3d.BYOLModel(**dict_args)
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

    # RELOAD MODELS IF RESUME TRUE
    if args.resume:
        state = torch.load(args.model_path)
        model.load_state_dict(state['model_state_dict'])

    # =============================================================================

    trainer = Trainer(run_num = args.run_num,
    				  model = model,
                      datamodule = dm,
                      grad_acc = args.grad_acc,
                      acc_bs = args.acc_bs,
                      max_epochs = args.max_epochs,
                      train_epochs = args.train_epochs,
                      lineval_epochs = args.lineval_epochs,
                      lineval_optim = args.lineval_optim,
                      lineval_lr = args.lineval_lr,
                      lineval_momentum = args.lineval_momentum,
                      lineval_wd = args.lineval_wd,
                      lineval_lr_schedule = args.lineval_lr_schedule,
                      finetune_epochs = args.finetune_epochs,
                      finetune_lr = args.finetune_lr,
                      finetune_momentum = args.finetune_momentum,
                      finetune_wd = args.finetune_wd,
                      finetune_lr_schedule = args.finetune_lr_schedule,
                      modelsavepath = args.modelsavepath,
                      modelsaveinterval = args.modelsaveinterval,
                      resume = args.resume,
                      model_path = args.model_path,
                      ft_fc_lr = args.ft_fc_lr,
                      lowest_val_eval = args.lowest_val_eval
                      )

    # =============================================================================


    trainer.fit()

    # =============================================================================
    #BUILD A MODEL FOR THE DOWNSTREAM TASK
    ds_model = ClassificationModel(args.base_encoder_name,
                                   dm.num_classes, 
                                   args.data_dims,
                                   classification_type = 'binary').to('cuda:0')

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
    fine_tune_metrics = trainer.fine_tune(ds_model)

    # =============================================================================

    metrics_dict = {**fine_tune_metrics} # {**knn_eval_metrics, **lin_eval_metrics, 
    # print(metrics_dict)
    # print(dict_args)

    # =============================================================================

    trainer.writer.add_hparams(dict_args, metrics_dict,
                                run_name = '_'.join([args.model_name,
                                                    args.base_encoder_name,
                                                    args.optimizer,'lr',str(args.lr),
                                                    'bs',str(args.pretrain_batch_size),
                                                    'ep',str(args.train_epochs), date]))

    # =============================================================================

    trainer.writer.close()

    # =============================================================================