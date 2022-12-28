import torch
import torch.nn as nn
from typing import Any
from torch import Tensor
from torchvision import models
import torchmetrics

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()
        self.in_features = None
        self.out_features = None

    def forward(self, x: Tensor)-> Tensor:
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 use_bn: bool = False,
                 **kwargs: Any) -> None:
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear_layer = nn.Linear(self.in_features,
                                self.out_features,
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
            self.bn_layer = nn.BatchNorm1d(self.out_features)

    def forward(self,x: Tensor) -> Tensor:
        x = self.linear_layer(x)
        if self.use_bn:
            x = self.bn_layer(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features: int = None,
                 hidden_features: int = None,
                 out_features: int = None,
                 head_type: str = 'nonlinear',
                 num_layers: int = 2,
                 use_bn: bool = True,
                 **kwargs: Any) -> None:
        super(ProjectionHead,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type
        self.num_layers = num_layers
        self.use_bn = use_bn
        # FOR BYOL
        self.last_bn = kwargs['last_bn'] if kwargs.__contains__('last_bn') else self.use_bn

        if self.head_type == 'linear':
            self.mlp = LinearLayer(self.in_features,self.out_features, False, self.use_bn)
        elif self.head_type == 'nonlinear':
            self.layers = []
            self.layers.append(LinearLayer(self.in_features,self.hidden_features, use_bias = True, use_bn = self.use_bn))
            self.layers.append(nn.ReLU())
            for i in range(self.num_layers-2):
                self.layers.append(LinearLayer(self.hidden_features,self.hidden_features, use_bias = True, use_bn = self.use_bn))
                self.layers.append(nn.ReLU())
            self.layers.append(LinearLayer(self.hidden_features,self.out_features, use_bias = True, use_bn = self.last_bn))

            self.mlp = nn.Sequential(*self.layers)

    def forward(self,x: Tensor) -> Tensor:
        x = self.mlp(x)
        return x

class ClassificationModel(nn.Module):
    def __init__(self,
                 base_encoder_name: str,
                 num_classes: int,
                 data_dims: str,
                 classification_type: str = 'multi-class',
                 eval_type: str = 'knn',
                 **kwargs: Any) -> None:
        super(ClassificationModel, self).__init__()
        self.base_encoder_name = base_encoder_name
        self.num_classes = num_classes
        self.data_dims = data_dims
        self.classification_type = classification_type
        self.eval_type = eval_type

        self.base_enc_out = None
        if self.num_classes == 2 and self.classification_type == 'binary':
            num_classes = 1
        elif self.num_classes == 2 and self.classification_type == 'multi-label':
            num_classes = 2
        else:
            num_classes = self.num_classes
        
        self.base_encoder = models.video.r3d_18(pretrained = False)
        self.base_encoder.fc = nn.Linear(512, num_classes)
        
        # if self.data_dims.split('x')[0] == '32':
        #     self.base_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        #     self.base_encoder.maxpool = Identity()

        # if self.eval_type == 'knn':
        #     prev_layer_out_features = self.base_encoder.fc.in_features
        #     self.base_encoder.fc = Identity()
        #     self.base_encoder.fc.in_features = prev_layer_out_features
        #     self.base_encoder.fc.out_features = prev_layer_out_features
        
        self.bin_pos_wts = kwargs['bin_pos_wts'] if kwargs.__contains__('bin_pos_wts') else torch.Tensor([1.0]*num_classes)
        if self.classification_type == 'multiclass':
            self.criterion = nn.CrossEntropyLoss(weight = self.bin_pos_wts)
        elif self.classification_type in ['binary','multilabel']:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight = self.bin_pos_wts)

        #if stage == 'lineval':
        #    for p in self.base_encoder
        #if self.eval_type == 'knn':
        self.fhooks = []
        self.fhooks.append(getattr(self.base_encoder,'avgpool').register_forward_hook(self.forward_hook('avgpool')))

        self.train_acc = torchmetrics.Accuracy(task=self.classification_type)
        self.valid_acc = torchmetrics.Accuracy(task=self.classification_type)
        self.test_acc = torchmetrics.Accuracy(task=self.classification_type)

        self.valid_auc = torchmetrics.AUROC(task=self.classification_type)
        self.test_auc = torchmetrics.AUROC(task=self.classification_type)

    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.base_enc_out = output.squeeze()
        return hook

    def forward(self, x:Tensor) -> Tensor:
        return self.base_encoder(x)

    def training_step(self, batch, batch_idx) : #, knn = True):
        x, y = [b.cuda(non_blocking = True) for b in batch]
        # pass throught net
        z = self.base_encoder(x)
        
        if self.classification_type != 'multi-class':
            y = y.to(dtype = torch.float)
        if self.classification_type == 'binary':
            y = y.view((-1,1))
        
        # if not knn:
        loss = self.criterion(z,y)

        if self.classification_type == 'multi-class':
            preds = torch.exp(z.cpu())/torch.sum(torch.exp(z.cpu()), dim = 1, keepdims = True)
            _, preds = torch.max(preds, dim = 1)
        else:
            preds = (z.cpu().data >= 0.5)
    
        acc = (preds == y.cpu()).to(dtype = torch.float64).mean().item()
        self.log('train_loss_downstream',loss, on_epoch = True, logger = True)
        self.log('train_acc_downstream',acc, on_epoch = True, logger = True)

        return loss #, acc, z.cpu(), y.cpu()
        # else:
            # return self.base_enc_out.cpu(), y.cpu()

    def validation_step(self, batch, batch_idx) : #, knn = True):
        x, y = [b.cuda(non_blocking = True) for b in batch]
        # pass throught net
        z = self.base_encoder(x)
        
        if self.classification_type != 'multiclass':
            y = y.to(dtype = torch.float)
        if self.classification_type == 'binary':
            y = y.view((-1,1))
        
        # if not knn:
        loss = self.criterion(z,y)

        if self.classification_type == 'multiclass':
            preds = torch.exp(z.cpu())/torch.sum(torch.exp(z.cpu()), dim = 1, keepdims = True)
            _, preds = torch.max(preds, dim = 1)
        else:
            preds = (z.cpu().data >= 0.5)
    
        acc = (preds == y.cpu()).to(dtype = torch.float64).mean().item()

        self.valid_acc.update(z, y)
        self.valid_auc.update(z, y)

        self.log('valid_loss_downstream',loss, on_step = True, on_epoch = False, logger = True)
        self.log('valid_acc_downstream',acc, on_step = True, on_epoch = False, logger = True)

        return loss #, acc, z.cpu(), y.cpu()
        # else:
            # return self.base_enc_out.cpu(), y.cpu()

    def testing_step(self, batch, batch_idx) : #, knn = True):
        x, y = [b.cuda(non_blocking = True) for b in batch]
        # pass throught net
        z = self.base_encoder(x)
        
        if self.classification_type != 'multi-class':
            y = y.to(dtype = torch.float)
        if self.classification_type == 'binary':
            y = y.view((-1,1))
        
        # if not knn:
        loss = self.criterion(z,y)

        if self.classification_type == 'multi-class':
            preds = torch.exp(z.cpu())/torch.sum(torch.exp(z.cpu()), dim = 1, keepdims = True)
            _, preds = torch.max(preds, dim = 1)
        else:
            preds = (z.cpu().data >= 0.5)
    
        acc = (preds == y.cpu()).to(dtype = torch.float64).mean().item()

        self.test_acc.update(z, y)
        self.test_auc.update(z, y)

        self.log('test_loss_downstream',loss, on_step = True, on_epoch = False, logger = True)
        self.log('test_acc_downstream',acc, on_step = True, on_epoch = False, logger = True)

        return loss #, acc, z.cpu(), y.cpu()
        # else:
            # return self.base_enc_out.cpu(), y.cpu()

    def training_epoch_end(self, outputs):
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.valid_acc.compute())
        self.valid_acc.reset()
        self.log('valid_auc_epoch', self.valid_auc.compute())
        self.valid_auc.reset()


    def configure_optimizers(self):
        params = list(self.backbone.parameters()) #\
            # + list(self.projection_head.parameters()) \
            # + list(self.prediction_head.parameters())
        optim = torch.optim.SGD(
            params, 
            lr=6e-2,# * lr_factor,
            momentum=0.9, 
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

