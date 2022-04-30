import os
import argparse #ArgumentParser, BooleanOptionalAction
from tabulate import tabulate
from datetime import datetime

import simclr, moco, simsiam, byol, barlow_twins, cumi, mdmi
from dataloader_modules import CIFAR10ArrayDataModule, CIFAR100DataModule, STL10DataModule, TinyImageNetDataModule
from trainer import Trainer
from utils import run_command
from model_utils import ClassificationModel
from model_transforms import *

def main(args):

    date = datetime.now().strftime("%d-%m-%Y")
    dict_args = vars(args)
    args_table = [[k,v] for k,v in dict_args.items()]
    print(tabulate(args_table,headers = ["Argument","Value"], tablefmt = "pretty"))

    #print(args.download, type(args.download), 'True' if args.download == True else 'False')

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.modelsavepath):
        os.makedirs(args.modelsavepath)

    if args.dataset=='cifar10' or args.dataset=='cifar100':
        args.data_dims = '32x32' if args.data_dims is None else args.data_dims

    # if args.model_name == 'simclr':
    #     model = simclr.SimCLRModel(**dict_args)
    #     transforms = SimCLRTransform(0.5,int(args.data_dims.split('x')[0]))
    #     #checkpoint_callback = ModelCheckpoint(dirpath = args.modelsavepath,
    #     #                                      period = args.modelsaveinterval,
    #     #                                      filename = '_'.join([args.model_name,args.dataset,args.base_encoder_name,args.optimizer]))
    # elif args.model_name == 'mocov1' or args.model_name == 'mocov2':
    #     model = moco.MoCoModel(**dict_args)
    #     if args.model_name == 'mocov1':
    #         transforms = MoCov1Transform(int(args.data_dims.split('x')[0]))
    #     else:
    #         transforms = MoCov2Transform(int(args.data_dims.split('x')[0]))

    # elif args.model_name == 'simsiam':
    #     model = simsiam.SimSiamModel(**dict_args)
    #     transforms = SimCLRTransform(0.5,int(args.data_dims.split('x')[0]))

    # elif args.model_name == 'byol':
    #     model = byol.BYOLModel(**dict_args)
    #     transforms = BYOLTransform(int(args.data_dims.split('x')[0]))

    # elif args.model_name == 'barlow_twins':
    #     # if dict_args['bn_bias_lr'] is None:
    #     #     dict_args['bn_bias_lr'] = 0.0048
    #     # if dict_args['lambd'] is None:
    #     #     dict_args['lambd'] = 0.0051
    #     # if dict_args['ft_fc_lr'] is None:
    #     #     dict_args['ft_fc_lr'] = 0.5
    #     model = barlow_twins.BTModel(**dict_args)
    #     transforms = BTTransform(int(args.data_dims.split('x')[0]))

    # elif args.model_name == 'cumi':
    #     model = cumi.CUMIModel(**dict_args)
    #     transforms = SimCLRTransform(0.5, int(args.data_dims.split('x')[0]))

    # elif args.model_name == 'mdmi':
    #     model = mdmi.MDMIModel(**dict_args)
    #     transforms = SimCLRTransform(0.5, int(args.data_dims.split('x')[0]))


    if args.dataset == 'cifar10':
        dm = CIFAR10ArrayDataModule(args.pretrain_batch_size,
                               args.other_batch_size,
                               args.download,
                               args.dataset_path,
                               transforms
                               )
    elif args.dataset == 'cifar100':
        dm = CIFAR100DataModule(args.pretrain_batch_size,
                               args.other_batch_size,
                               args.download,
                               args.dataset_path,
                               transforms
                               )
    elif args.dataset == 'stl10':
        dm = STL10DataModule(args.pretrain_batch_size,
                           args.other_batch_size,
                           args.download,
                           args.dataset_path,
                           transforms
                           )

    elif args.dataset == 'tinyimagenet':
        dm = TinyImageNetDataModule(args.pretrain_batch_size,
                                    args.other_batch_size,
                                    args.download,
                                    args.dataset_path,
                                    transforms
                                    )

    trainer = Trainer(None,
                      dm,
                      max_epochs = None, #args.max_epochs,
                      train_epochs = None, #args.train_epochs,
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
                      ft_fc_lr = args.ft_fc_lr
                      )


    #trainer.load_model_for_eval()
    
    #BUILD A MODEL FOR THE DOWNSTREAM TASK
    ds_model = ClassificationModel(args.base_encoder_name,
                                   dm.num_classes, 
                                   args.data_dims).to('cuda:0')
    #KNN EVALUATION
    knn_eval_metrics = trainer.knn_eval(ds_model, 
                                        fracs = args.knn_fracs, 
                                        k = args.num_neighbours, 
                                        weights = args.knn_wt_type, 
                                        algorithm = args.knn_algo_type, 
                                        metric = args.knn_metric_type)
    # LINEAR EVALUATION
    #ds_model = ClassificationModel('resnet18',dm.num_classes).to('cuda:0')
    lin_eval_metrics = trainer.linear_eval(ds_model, net_model_path = args.model_path, mode = 'infer')
    #FINE TUNING
    #fine_tune_metrics = trainer.fine_tune(ds_model)

    metrics_dict = {**knn_eval_metrics, **lin_eval_metrics} #, **fine_tune_metrics}
    # print(metrics_dict)
    # print(dict_args)
    trainer.writer.add_hparams(dict_args, metrics_dict,
                                run_name = '_'.join([args.model_name,
                                                    args.base_encoder_name,
                                                    'inference', date]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus',type=int,default=1)
    parser.add_argument('--pretrain',action='store_false',help = 'Stage of learning')
    parser.add_argument('--model_name',type=str,default='simclr',help = 'Name of Algorithm or Framework')
    parser.add_argument('--base_encoder_name',type=str,default='resnet50')
    # parser.add_argument('--projector_type',type=str,default='nonlinear')
    # parser.add_argument('--proj_num_layers',type=int,default=2)
    # parser.add_argument('--projector_hid_dim',type=int,default=2048)
    # parser.add_argument('--projector_out_dim',type=int,default=128)
    # parser.add_argument('--proj_use_bn', action='store_false')
    # parser.add_argument('--proj_last_bn', action='store_true')
    # parser.add_argument('--predictor_type',type=str,default='nonlinear')
    # parser.add_argument('--pred_num_layers',type=int,default=None)
    # parser.add_argument('--predictor_hid_dim',type=int,default=None)
    # parser.add_argument('--predictor_out_dim',type=int,default=None)
    # parser.add_argument('--pred_use_bn',action='store_false')
    # parser.add_argument('--pred_last_bn',action='store_true')
    # parser.add_argument('--K',type=int,default=65536)
    # parser.add_argument('--m',type=float,default=0.99)
    # parser.add_argument('--optimizer',type=str,default='sgd')
    # parser.add_argument('--lr',type=float,default=0.3)
    # parser.add_argument('--lr_scheduler',type=str,default='lwca')
    # parser.add_argument('--momentum',type=float, default = 0.9)
    # parser.add_argument('--weight_decay',type=float,default=1e-6)
    # parser.add_argument('--temperature',type=float,default=0.5)
    # parser.add_argument('--color_distortion_strength',type=float,default=0.5)
    # parser.add_argument('--train_epochs',type=int,default=100)
    # parser.add_argument('--warmup_epochs',type=int,default=10)
    # parser.add_argument('--max_epochs',type=int,default=1000)
    # parser.add_argument('--warmup_start_lr',type=float,default=1e-4)
    # parser.add_argument('--eta_min',type=float,default=1e-4)
    # parser.add_argument('--pretrain_batch_size',type=int,default=64)
    # parser.add_argument('--other_batch_size',type=int,default=32)
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
    # parser.add_argument('--bn_bias_lr',type=float, default = None)
    parser.add_argument('--ft_fc_lr',type=float, default = None)
    # parser.add_argument('--lambd',type=float, default = 1.0)
    # parser.add_argument('--mu',type=float, default = 1.0)
    # parser.add_argument('--nu',type=float, default = 1.0)

    # parser.add_argument('--n_temperature', type=float, default = None)
    # parser.add_argument('--lambda_loss', type=float, default = 1.0)

    # parser.add_argument('--step_size', type=int, default = 10)
    # parser.add_argument('--lr_gamma', type=float, default=0.1)
    # parser.add_argument('--lr_milestones', type=str, default=None)
    # parser.add_argument('--lr_factor', type=float, default=None)
    # parser.add_argument('--lr_total_iters', type=int, default=None)


    args = parser.parse_args()

    #SYSTEM INFORMATION
    command = ['nvidia-smi']
    run_command(command)
    #more system INFORMATION
    command = ['python', '-m', 'torch.utils.collect_env']
    run_command(command)

    main(args)
