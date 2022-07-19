from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

from model_utils import Identity, ProjectionHead
from torch.optim import SGD
from schedulers import LinearWarmupCosineAnnealingLR
from lars import LARS
from losses import BYOLLoss
from utils import seed_everything

import matplotlib.pyplot as plt

class BYOLNet(nn.Module):
    def __init__(self,
                 base_encoder_name: str = 'r3d_18',
                 projector_type: str = 'nonlinear',
                 proj_num_layers: int = 2,
                 projector_hid_dim: int = 4096,
                 projector_out_dim: int = 256,
                 proj_use_bn: bool = True,
                 proj_last_bn: bool = False,
                 predictor_type: str = 'nonlinear',
                 pred_num_layers: int = 2,
                 predictor_hid_dim: int = 4096,
                 predictor_out_dim: int = 256,
                 pred_use_bn: bool = True,
                 pred_last_bn: bool = False,
                 data_dims: str = '224x224',
                 online: bool = True) -> None:
        super().__init__()
        self.base_encoder_name = base_encoder_name
        self.projector_type = projector_type
        self.proj_num_layers = proj_num_layers
        self.projector_hid_dim = projector_hid_dim
        self.projector_out_dim = projector_out_dim
        self.proj_use_bn = proj_use_bn
        self.proj_last_bn = proj_last_bn
        self.predictor_type = predictor_type
        self.pred_num_layers = pred_num_layers
        self.predictor_hid_dim = predictor_hid_dim
        self.predictor_out_dim = predictor_out_dim
        self.pred_use_bn = pred_use_bn
        self.pred_last_bn = pred_last_bn
        self.data_dims = data_dims
        self.online = online

        self.base_encoder = models.video.r3d_18()
        dim_infeat = self.base_encoder.fc.weight.shape[1]
        self.base_encoder.fc = Identity()
        if self.data_dims.split('x')[0] == '32':
            self.base_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
            self.base_encoder.maxpool = Identity()

        for p in self.base_encoder.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(in_features = dim_infeat,
                                        hidden_features = self.projector_hid_dim,
                                        out_features = self.projector_out_dim,
                                        head_type = self.projector_type,
                                        num_layers = 2,
                                        use_bn = True,
                                        last_bn = False)
        self.predictor = ProjectionHead(in_features = self.projector_out_dim,
                                        hidden_features = self.projector_hid_dim,
                                        out_features = self.projector_out_dim,
                                        head_type = self.projector_type,
                                        num_layers = 2,
                                        use_bn = True,
                                        last_bn = False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.base_encoder(x)
        z = self.projector(x)
        if self.online:
            z = self.predictor(z)
        return x, z

class BYOLModel(nn.Module):
    def __init__(self,
                 base_encoder_name: str = 'resnet50',
                 data_dims: str = '224x224',
                 optim: str = 'lars',
                 lr: float = 0.2,
                 lr_scheduler: str = 'lwca',
                 momentum: float = 0.9,
                 weight_decay: float = 1.5e-6,
                 warmup_epochs: int = 10,
                 max_epochs: int = 200,
                 warmup_start_lr: float = 1e-5,
                 eta_min: float = 1e-5,
                 pretrain_batch_size: int = 64,
                 ds_batch_size: int = 32,
                 temperature: float = 0.5,
                 projector_type: str = 'nonlinear',
                 proj_num_layers: int = 2,
                 projector_hid_dim: int = 4096,
                 projector_out_dim: int = 256,
                 proj_use_bn: bool = True,
                 proj_last_bn: bool = False,
                 predictor_type: str = 'nonlinear',
                 pred_num_layers: int = 2,
                 predictor_hid_dim: int = 4096,
                 predictor_out_dim: int = 256,
                 pred_use_bn: bool = True,
                 pred_last_bn: bool = False,
                 m: float = 0.996,
                 **kwargs: Any) -> None:
        super().__init__()

        self.base_encoder_name = base_encoder_name
        self.data_dims = data_dims
        self.optim = optim
        self.pretrain_batch_size = pretrain_batch_size
        self.base_lr = 0.2
        self.lr = lr #min(self.base_lr*self.pretrain_batch_size/256, lr) if self.pretrain_batch_size >= 256 else self.base_lr
        self.lr_scheduler = lr_scheduler
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        self.ds_batch_size = ds_batch_size
        self.temperature = temperature

        self.projector_type = projector_type
        self.proj_num_layers = proj_num_layers
        self.projector_hid_dim = projector_hid_dim
        self.projector_out_dim = projector_out_dim
        self.proj_use_bn = proj_use_bn
        self.proj_last_bn = proj_last_bn
        self.predictor_type = predictor_type
        self.pred_num_layers = pred_num_layers
        self.predictor_hid_dim = predictor_hid_dim
        self.predictor_out_dim = predictor_out_dim
        self.pred_use_bn = pred_use_bn
        self.pred_last_bn = pred_last_bn

        self.base_m = m
        self.m = m
        self.global_step = 0
        self.max_training_steps = max_epochs

        self.kwargs = kwargs

        self.bn_bias_lr = self.kwargs['bn_bias_lr'] if self.kwargs['bn_bias_lr'] is not None else None

        self.net = BYOLNet(self.base_encoder_name,
                             self.projector_type,
                             self.proj_num_layers,
                             self.projector_hid_dim,
                             self.projector_out_dim,
                             self.proj_use_bn,
                             self.proj_last_bn,
                             self.predictor_type,
                             self.pred_num_layers,
                             self.predictor_hid_dim,
                             self.predictor_out_dim,
                             self.pred_use_bn,
                             self.pred_last_bn,
                             self.data_dims,
                             True).to('cuda:0')

        self.net_t = BYOLNet(self.base_encoder_name,
                             self.projector_type,
                             self.proj_num_layers,
                             self.projector_hid_dim,
                             self.projector_out_dim,
                             self.proj_use_bn,
                             self.proj_last_bn,
                             self.predictor_type,
                             self.pred_num_layers,
                             self.predictor_hid_dim,
                             self.predictor_out_dim,
                             self.pred_use_bn,
                             self.pred_last_bn,
                             self.data_dims,
                             False).to('cuda:0')

        self.criterion = BYOLLoss()

        for param_o, param_t in zip(self.net.parameters(), self.net_t.parameters()):
            param_t.data.copy_(param_o.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        self.optimizer, self.scheduler = self.configure_optimizers()

    @property
    def model_name(self) -> str:
        return 'byol'

    def update_target_encoder_momentum(self):
        self.m = 1 - (1-self.base_m)*(np.cos(np.pi*self.global_step/self.max_training_steps) + 1)/2
        self.global_step += 1

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the key encoder
        """
        self.update_target_encoder_momentum()
        for param_o, param_t in zip(self.net.parameters(), self.net_t.parameters()):
            param_t.data = param_t.data * self.m + param_o.data * (1. - self.m)

    def forward(self, im_o, im_t, update_mom = False):
        # compute query features
        e1, o1 = self.net(im_o)
        _, t1 = self.net(im_t)
        # queries: NxC
        # compute key features
        with torch.no_grad():  # no gradient to keys
            if update_mom:
                self._momentum_update_target_encoder()  # update the key encoder
            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, o2 = self.net_t(im_o)
            _, t2 = self.net_t(im_t)  # keys: NxC
            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return e1, (o1, t1), (o2, t2)

    def configure_optimizers(self):
        if self.bn_bias_lr is not None:
            param_weights = []
            param_weights_names = []
            param_biases = []
            param_biases_names = []
            param_names = []
            for name, param in self.net.named_parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                    param_biases_names.append(name)
                else:
                    param_weights.append(param)
                    param_weights_names.append(name)
                param_names.append(name)

            parameters = [{'params': param_weights, 'lr':self.lr, 'param_names':param_weights_names},
                          {'params': param_biases, 'lr':self.bn_bias_lr,'param_names':param_biases_names}]
        else:
            params = []
            param_names = []
            for n,p in self.net.named_parameters():
                params.append(p)
                param_names.append(n)
            parameters = [{'params':params,'param_names':param_names}]

        if self.optim == 'lars':
            optimizer = LARS(parameters,
                             lr = self.lr,
                             weight_decay = self.weight_decay,
                             exclude_from_weight_decay=["batch_normalization", "bias"] if self.bn_bias_lr is not None else [])
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(parameters,
                                        lr = self.lr,
                                        momentum = self.momentum,
                                        weight_decay = self.weight_decay)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(parameters,
                                         lr = self.lr,
                                         weight_decay = self.weight_decay)

        if self.lr_scheduler == 'lwca':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                      self.warmup_epochs,
                                                      self.max_epochs,
                                                      self.warmup_start_lr,
                                                      self.eta_min)
        elif self.lr_scheduler == 'nodec':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 
                                                            factor=1.0, 
                                                            total_iters=self.max_epochs, 
                                                            last_epoch=- 1, 
                                                            verbose=True)

        elif self.lr_scheduler == 'cadec':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   self.max_epochs, 
                                                                   eta_min=0, 
                                                                   last_epoch=- 1, 
                                                                   verbose=True)

        elif self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size = self.kwargs['step_size'] if self.kwargs['step_size'] is not None else 10,
                                                        gamma = self.kwargs['lr_gamma'] if self.kwargs['lr_gamma'] is not None else 0.1,
                                                        verbose = True)
        elif self.lr_scheduler == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones = self.kwargs['lr_milestones'] if self.kwargs['lr_milestones'] is not None else [int(0.6*self.max_epochs), int(0.8*self.max_epochs)],
                                                             gamma = self.kwargs['lr_gamma'] if self.kwargs['lr_gamma'] is not None else 0.1,
                                                             verbose = True)
        elif self.lr_scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                          start_factor = self.kwargs['lr_factor'] if self.kwargs['lr_factor'] is not None else 0.99, 
                                                          end_factor = 1.0, 
                                                          total_iters = self.kwargs['lr_total_iters'] if self.kwargs['lr_total_iters'] is not None else self.max_epochs, 
                                                          last_epoch = - 1, 
                                                          verbose = True)
        elif self.lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                               gamma = self.kwargs['lr_gamma'] if self.kwargs['lr_gamma'] is not None else 0.1,
                                                               last_epoch=- 1, 
                                                               verbose= True)


        return optimizer, scheduler

    def step(self, stage, batch, batch_idx, update_mom, summary_writer = None):
        x1, x2, y = [b.cuda(non_blocking = True) for b in batch]
        # plt.imshow(x1.cpu().numpy()[0,:,0,:,:].transpose(1,2,0)*0.1087 + 0.1593)
        # plt.show()
        #  pass throught net
        e1, (o1, t1), (o2, t2) = self(x1, x2, update_mom)
        # print(o1)
        #print(q.shape,k.shape)
        loss1 = self.criterion(o1, t2, batch_idx, summary_writer, stage)        
        loss2 = self.criterion(t1, o2, batch_idx, summary_writer, stage)
        loss = loss1 + loss2
        #print(loss)
        #  self.log('train_loss_ssl',loss, on_epoch = True, logger = True)
        if stage == 'train':
            return loss
        else:
            return e1.cpu().numpy(), y.cpu().numpy(), loss #.cpu().item()
