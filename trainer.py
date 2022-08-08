import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_utils import ClassificationModel
from typing import Type, Any
from datetime import datetime
from utils import save_model, plot_features
from perf_metrics import get_performance_metrics

from sklearn.neighbors import KNeighborsClassifier

class Trainer(nn.Module):
    def __init__(self,
                 run_num: str,
                 model: nn.Module,
                 datamodule: nn.Module,
                 grad_acc : bool = False,
                 acc_bs: int = 2,
                 train_epochs: int = 100,
                 lineval_epochs: int = 100,
                 lineval_optim: str = 'sgd',
                 lineval_lr: float = 0.01,
                 lineval_momentum: float = 0.9,
                 lineval_wd: float = 0.0,
                 lineval_lr_schedule: str = 'steplr',
                 finetune_epochs: int = 100,
                 finetune_optim: str = 'sgd',
                 finetune_lr: float = 0.01,
                 finetune_momentum: float = 0.9,
                 finetune_wd: float = 0.0,
                 finetune_lr_schedule: str = 'steplr',
                 max_epochs: int = 1000,
                 modelsavepath: str = os.getcwd(),
                 modelsaveinterval: int = 1,
                 resume: bool = False,
                 model_path: str = None,
                 show_valid_metrics: bool = True,
                 lowest_val_eval: bool = False,
                 **kwargs: Any) -> None:
        super().__init__()

        self.model = model
        self.datamodule = datamodule
        self.train_epochs = train_epochs
        self.lineval_epochs = lineval_epochs
        self.lineval_optim = lineval_optim
        self.lineval_lr = lineval_lr
        self.lineval_momentum = lineval_momentum
        self.lineval_wd = lineval_wd
        self.lineval_lr_schedule = lineval_lr_schedule
        self.finetune_epochs = finetune_epochs
        self.finetune_optim = finetune_optim
        self.finetune_lr = finetune_lr
        self.finetune_momentum = finetune_momentum
        self.finetune_wd = finetune_wd
        self.finetune_lr_schedule = finetune_lr_schedule
        self.max_epochs = max_epochs
        self.modelsavepath = modelsavepath
        self.modelsaveinterval = modelsaveinterval
        self.resume = resume
        self.model_path = model_path
        self.model_name = self.model.model_name
        self.show_valid_metrics = show_valid_metrics
        self.lowest_val_eval = lowest_val_eval

        self.grad_acc = grad_acc
        self.batch_size = self.model.pretrain_batch_size
        self.acc_bs = acc_bs
        self.acc_iter = self.batch_size // self.acc_bs if self.grad_acc else 1
        
        self.ds_num_classes = None

        self.ft_fc_lr = kwargs['ft_fc_lr'] if kwargs['ft_fc_lr'] is not None else self.finetune_lr
        
        self.datestr = datetime.today().strftime("%d-%m-%y-%H-%M-%S")

        self.run_num = '_'.join([datetime.today().strftime("%d-%m-%y"), run_num])

        # self.datamodule.prepare_data()

        self.writer = SummaryWriter('/'.join(['runs','_'.join([self.model_name,
                                                               self.model.base_encoder_name,
                                                               self.run_num])]))

    def fit(self):
        if self.resume:
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.model.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.model.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            start_epoch = self.model.scheduler.last_epoch
            print("Resuming Training...Starting Epoch: ",start_epoch)
        else:
            start_epoch = 0

        #saving the final model
        if self.resume == True and start_epoch == self.train_epochs:
            epoch = start_epoch
            self.final_net_save_path = self.model_path
            
        self.datamodule.setup(stage = 'train',pretrain = True)
        self.datamodule.setup(stage = 'valid',pretrain = True)
        self.train_loader = self.datamodule.train_dataloader(True)
        self.valid_loader = self.datamodule.valid_dataloader(True)
        if hasattr(self.model, 'max_training_steps'):
            self.model.max_training_steps = self.max_epochs*(len(self.train_loader))
        #self.optimizer, self.scheduler = self.model.configure_optimizers()
        self.train_losses, self.valid_losses = np.array([]), np.array([])
        self.min_val_loss = None

        for epoch in range(start_epoch, self.train_epochs):
            print("\nEpoch {}".format(epoch+1), flush = True)
            train_epoch_loss = self.train_epoch(self.model, self.train_loader, self.model.optimizer, epoch, self.writer)
            self.train_losses = np.append(self.train_losses, train_epoch_loss)
            self.writer.add_scalar('Pretrain/Loss/train',train_epoch_loss, epoch)
            self.model.scheduler.step()
            features, labels, val_epoch_loss = self.valid_epoch(self.model, self.valid_loader, epoch, self.writer)
            self.writer.add_scalar('Pretrain/Loss/valid',val_epoch_loss, epoch)
            if self.min_val_loss is None or val_epoch_loss < self.min_val_loss:
                self.min_val_loss = val_epoch_loss
                self.lowest_val_model_path = save_model(self.model,
                                                       epoch+1,
                                                       self.modelsavepath,
                                                       '_'.join([self.model.model_name,self.run_num,'lowest_val.pt']))
            self.valid_losses = np.append(self.valid_losses, val_epoch_loss)
            #fig = plot_metrics(self.train_losses, self.valid_losses, 'Loss')
            #self.writer.add_figure('Pretrain/metrics',fig,epoch)
            if (epoch+1) % self.modelsaveinterval == 0:
                self.final_model_save_path = save_model(self.model,
                                                       epoch+1,
                                                       self.modelsavepath,
                                                       '_'.join([self.model.model_name,
                                                                self.run_num, '{}.pt']))
                print(f"Model at epoch {epoch+1} saved at {self.final_model_save_path}")
                fig = plot_features(features, labels, 2 if self.datamodule.num_classes==1 else self.datamodule.num_classes, epoch)
                self.writer.add_figure('Pretrain/TSNE-Features',fig,epoch)

        #saving the final model
        if start_epoch < self.train_epochs:
            #epoch = start_epoch
            self.final_model_save_path = save_model(self.model, 
                                                    epoch+1, 
                                                    self.modelsavepath, 
                                                    '_'.join([self.model.model_name,self.run_num,'final.pt']))
            print(f"Final Model at epoch {epoch+1} saved at {self.final_model_save_path}")

            #SAVING ONLY THE ENCODER
            self.final_net_save_path = '/'.join([self.modelsavepath,
                                                '_'.join([self.model.model_name,
                                                          self.run_num,
                                                          'final_net.pt'])])
            torch.save(self.model.net.state_dict(), self.final_net_save_path)
            print(f"Encoder of Final Model at epoch {epoch+1} saved at {self.final_net_save_path}")

    def load_model_for_eval(self):
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.model.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        start_epoch = self.model.scheduler.last_epoch
        print("Epoch State: ",start_epoch)
        
        self.datamodule.setup(stage = 'valid',pretrain = True)
        
        self.valid_loader = self.datamodule.valid_dataloader(True)
        if hasattr(self.model, 'max_training_steps'):
            self.model.max_training_steps = self.max_epochs*(len(self.train_loader))
        
        self.train_losses, self.valid_losses = np.array([]), np.array([])
        self.min_val_loss = None

        for epoch in range(1):
            print("\nExtracting Features...", flush = True)
            
            features, labels, val_epoch_loss = self.valid_epoch(self.model, self.valid_loader)
            self.writer.add_scalar('Inference/Loss/valid',val_epoch_loss, epoch)
            
            fig = plot_features(features, labels, self.datamodule.num_classes, epoch)
            self.writer.add_figure('Inference/TSNE-Features',fig,epoch)

    def linear_eval(self,
                    dsmodel,
                    patience: int = 5,
                    net_model_path: str = None,
                    mode: str = 'train'
                   ) -> Any:

        '''
            other_metrics: Dictionary {'metric1_name' : metric1_function,
                                        'metric2_name' : metric2_function}

        '''
        print("::::::::::::::::::LINEAR EVALUATION INFERENCE::::::::::::::::::")
        if mode == 'train':
            if net_model_path is not None:
                dsmodel.load_state_dict(torch.load(net_model_path), strict = False)
            else:
                if self.lowest_val_eval:
                    dsmodel.load_state_dict(torch.load(self.lowest_val_model_path), strict = False)
                else:
                    dsmodel.load_state_dict(torch.load(self.final_net_save_path), strict = False)

            metrics = self.train_classifier('linear_eval',
                                            mode,
                                            dsmodel,
                                            True, 1.0,
                                            self.lineval_epochs,
                                            patience,
                                            self.lineval_optim,
                                            self.lineval_lr_schedule
                                            )
        else:
            if net_model_path is not None:
                # For evaluation the saved model will not have separate model_state_dict
                dsmodel.load_state_dict(torch.load(net_model_path), strict = False)
            else:
                dsmodel.load_state_dict(torch.load(self.model_path), strict = False)

            metrics = self.infer_classifier('linear_eval',
                                            mode,
                                            dsmodel,
                                            True, 1.0
                                            )
        return metrics

    def fine_tune(self,
                  dsmodel,
                  patience: int = 5,
                  net_model_path: str = None,
                  mode: str = 'train'
                 ) -> Any:
        metrics_dict = {}
        for fracs in [1.0]:

            if mode == 'train':
                print(":::::::::::::Semi-Supervised Fine-Tuning for {frac:.5f}% of Training Data::::::::::::".format(frac=fracs*100), flush = True)
                # LOAD FINAL MODEL
                if net_model_path is not None:
                    dsmodel.load_state_dict(torch.load(net_model_path), strict = False)
                else:
                    if self.lowest_val_eval:
                        dsmodel.load_state_dict(torch.load(self.lowest_val_model_path), strict = False)
                    else:
                        dsmodel.load_state_dict(torch.load(self.final_net_save_path), strict = False)

                metrics = self.train_classifier('fine_tune',
                                                'train',
                                              dsmodel,
                                              False, fracs,
                                              self.finetune_epochs,
                                              patience,
                                              self.finetune_optim,
                                              self.finetune_lr_schedule
                                             )
                metrics_dict = {**metrics_dict, **metrics}
            else:
                print("::::::Inference on Semi-Supervised Fine-Tuned Model for {frac:.5f}% of Training Data:::::".format(frac=fracs*100), flush = True)
                #LOAD FINAL MODEL
                if net_model_path is not None:
                    dsmodel.load_state_dict(torch.load(net_model_path), strict = False)
                else:
                    dsmodel.load_state_dict(torch.load(self.model_path), strict = False)

                metrics = self.infer_classifier('fine_tune',
                                                mode,
                                                dsmodel,
                                                False, fracs
                                               )
                metrics_dict = {**metrics_dict, **metrics}

        return metrics_dict

    def train_epoch(self,
                    model: nn.Module,
                    dataloader: nn.Module,
                    optimizer: nn.Module,
                    epoch: int,
                    summary_writer: Any) -> int:
        model.train()
        train_losses = 0
        with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as tepoch:
            optimizer.zero_grad(set_to_none = True)
            for step, batch in enumerate(tepoch):
                train_loss = model.step('train', batch, epoch*len(dataloader)+step, True if (step==0 or step%self.acc_iter==0) else False, summary_writer)/self.acc_iter
                #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                train_loss.backward()
                train_losses += train_loss.item()
                #print(train_losses)
                tepoch.set_postfix(loss = self.acc_iter*train_loss.item())
                if ((step+1)%self.acc_iter==0 or (step+1)==len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none = True)
        train_losses = train_losses/(step+1)
        return train_losses

    def valid_epoch(self,
                    model: nn.Module,
                    dataloader: nn.Module,
                    epoch: int,
                    summary_writer: Any) -> int:
        model.eval()
        valid_losses = 0
        features = np.array([]).reshape((0,self.model.net.projector.in_features)) #_out_dim if not hasattr(self.model,'predictor_out_dim') or self.model.predictor_out_dim is None else self.model.predictor_out_dim))
        labels = np.array([])
        with torch.no_grad():
            with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as vepoch:
                for step, batch in enumerate(vepoch):
                    feats, label, valid_loss = model.step('valid',batch, epoch*len(dataloader)+step, summary_writer)
                    features = np.append(features, feats, axis = 0)
                    labels = np.append(labels, label)
                    #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                    valid_losses += valid_loss.item()
                    vepoch.set_postfix(loss = valid_loss.item())
            valid_losses = valid_losses/(step+1)
        return features, labels, valid_losses

    def train_classifier(self,
                       stage: str,
                       mode: str,
                       model: nn.Module,
                       linear_eval: bool,
                       fracs: float = 1.0,
                       ds_epochs: int = 100,
                       patience: int = 5,
                       optim: str = 'sgd',
                       scheduler: str = 'steplr'
                      ) -> None:

        stage = stage
        mode = mode
        model = model
        linear_eval = linear_eval
        #for p in model.parameters():
        #    p.requires_grad = False
        #if not linear_eval:
        #    for p in model.net.base_encoder.parameters():
        #        p.requires_grad = True

        fracs = fracs
        ds_epochs = ds_epochs
        patience = patience
        counter = 0

        optim = optim
        scheduler = scheduler

        lr = self.lineval_lr if linear_eval else self.finetune_lr
        momentum = self.lineval_momentum if linear_eval else self.finetune_momentum

        if linear_eval:
            for n,p in model.named_parameters():
                if 'fc' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

        self.evaluation_model = model #
        self.datamodule.setup(stage = 'train',pretrain = False, fracs = fracs)
        self.datamodule.setup(stage = 'valid',pretrain = False)
        self.train_loader = self.datamodule.train_dataloader(False)
        self.valid_loader = self.datamodule.valid_dataloader(False)
        self.datamodule.setup(stage = 'test',pretrain = False)
        self.test_loader = self.datamodule.test_dataloader(False)

        # setting different learning rates for differetn parts of the model
        if not linear_eval:
            params_encoder = []
            params_fc = []
            encoder_lr = self.finetune_lr
            fc_lr = self.ft_fc_lr
            for n,p in self.evaluation_model.named_parameters():
                if 'fc' not in n:
                    params_encoder.append(p)
                else:
                    params_fc.append(p)
            enc_params = {'params':params_encoder,'lr':encoder_lr}
            fc_params = {'params':params_fc, 'lr':fc_lr}
            parameters = [enc_params, fc_params]
        else:
            parameters = [p for p in self.evaluation_model.parameters() if p.requires_grad]

        if optim == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr = lr, momentum = momentum)
        if optim == 'adam':
            optimizer = torch.optim.Adam(parameters, lr = lr)
        if optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(parameters, lr = lr)

        # ======== ONLY FOR BINARY AND MULTILABEL CLASSIFICATION FOR IMBALNACED DATASETS
        self.evaluation_model.bin_pos_wts = self.datamodule.traingen.bin_pos_wts
        if self.evaluation_model.classification_type in ['binary','multi-label'] and not self.datamodule.dsoversample:
            self.evaluation_model.criterion = nn.BCEWithLogitsLoss(pos_weight = self.bin_pos_wts)
        elif self.evaluation_model.classification_type == 'multi-class' and not self.datamodule.dsoversample:
            self.evaluation_model.criterion = nn.CrossEntropyLoss(weight = self.bin_pos_wts)

        #CHANGE SCHEDULER LATER
        if scheduler == 'steplr':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=1,
                                                           gamma=0.98,
                                                           last_epoch=-1,
                                                           verbose = True)
        elif scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max = ds_epochs,
                                                                       last_epoch=-1,
                                                                       verbose = True)
        elif scheduler == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones = [int(0.6*ds_epochs), int(0.8*ds_epochs)],
                                                                gamma = 0.1,
                                                                verbose = True)
        self.train_losses, self.valid_losses = np.array([]), np.array([])
        self.train_accuracy, self.valid_accuracy = np.array([]), np.array([])
        
        if self.evaluation_model.classification_type != 'multi-class':
            if self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'binary':
                self.ds_num_classes = 1
            elif self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'multi-label':
                self.ds_num_classes = 2
        else:
            self.ds_num_classes = self.datamodule.num_classes

        for epoch in range(ds_epochs):
            print("\nEpoch {}".format(epoch+1), flush = True)
            train_epoch_loss, train_epoch_accuracy = self.train_ds_epoch(self.evaluation_model,
                                                                        self.train_loader,
                                                                        optimizer)

            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Loss','train']),train_epoch_loss,epoch)
            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Accuracy','train']),train_epoch_accuracy,epoch)
            self.train_losses = np.append(self.train_losses, train_epoch_loss)
            self.train_accuracy = np.append(self.train_accuracy, train_epoch_accuracy)
            lr_scheduler.step()
            val_epoch_loss, val_epoch_accuracy, preds, gts = self.valid_ds_epoch(self.evaluation_model,self.valid_loader)
            self.valid_losses = np.append(self.valid_losses, val_epoch_loss)
            self.valid_accuracy = np.append(self.valid_accuracy, val_epoch_accuracy)

            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Loss','valid']),val_epoch_loss,epoch)
            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Accuracy','valid']),val_epoch_accuracy, epoch)
            print("\nTrain Accuracy : {acc:.5f}, Train Loss : {loss:.5f}".format(acc = train_epoch_accuracy, loss = train_epoch_loss), flush = True)
            print("\nValid Accuracy : {acc:.5f}, Valid Loss : {loss:.5f}".format(acc = val_epoch_accuracy, loss = val_epoch_loss), flush = True)
            
            if self.show_valid_metrics:
                val_perf_metrics = get_performance_metrics(gts, preds, [str(c) for c in list(range(self.ds_num_classes))])
                print(val_perf_metrics)

            if val_epoch_loss <= min(self.valid_losses):
                counter = 0
                if linear_eval:
                    self.dsfilepath = '/'.join([self.modelsavepath,'_'.join([self.model_name, 'linear_eval', self.datestr, '.pt'])])
                    torch.save(self.evaluation_model.state_dict(), self.dsfilepath)
                else:
                    self.dsfilepath = '/'.join([self.modelsavepath,'_'.join([self.model_name, 'fine_tune', str(fracs).replace('.','p'), self.datestr, '.pt'])])
                    torch.save(self.evaluation_model.state_dict(), self.dsfilepath)
            else:
                counter+=1
                # if counter>patience:
                #     print("Stopping Early. No more patience left.")
                #     break

        #fig = plot_metrics(self.train_losses, self.valid_losses, 'Loss')
        #fig = plot_metrics(self.train_accuracy, self.valid_accuracy, 'Accuracy')
        ## LOADING THE BEST MODEL
        self.evaluation_model.load_state_dict(torch.load(self.dsfilepath))
        test_loss, test_acc, preds, gts = self.valid_ds_epoch(self.evaluation_model, self.test_loader)

        preds = preds.reshape((-1,5)).mean(axis = 1, keepdims = True)
        gts = gts.reshape((-1,5)).mean(axis = 1, keepdims = True)
        
        print(':::::Saving Predictions and One Hot Ground Truth data to \'.npy\' files:::::::')
        np.save('test_set_preds.npy', preds)
        np.save('test_set_onehot_gt.npy', gts)
        
        print(':::::::::::::::::::::::Class-wise Performance Metrics:::::::::::::::::::::::::')
        if self.evaluation_model.classification_type != 'multi-class':
            test_perf_metrics = get_performance_metrics(gts, preds, [str(c) for c in list(range(self.ds_num_classes))])
            print(test_perf_metrics)
        else:
            print("\nTest Accuracy : {acc:.5f}, Test Loss : {loss:.5f}".format(acc = test_acc, loss = test_loss), flush = True)
        
        min_ind = np.argmin(self.valid_losses)
        val_loss_min_ind = self.valid_losses[min_ind]
        val_acc_min_ind = self.valid_accuracy[min_ind]
        return {'_'.join([stage,mode,str(fracs).replace('.','p'),'val_loss']):val_loss_min_ind,
                '_'.join([stage,mode,str(fracs).replace('.','p'),'val_acc']):val_acc_min_ind,
                '_'.join([stage,mode,str(fracs).replace('.','p'),'test_loss']):test_loss,
                '_'.join([stage,mode,str(fracs).replace('.','p'),'test_acc']):test_acc}

    def infer_classifier(self,
                       stage: str,
                       mode: str,
                       model: nn.Module,
                       linear_eval: str,
                       fracs: float = 1.0
                      ) -> None:

        stage = stage
        mode = mode
        model = model
        linear_eval = linear_eval
        fracs = fracs

        if linear_eval:
            for n,p in model.named_parameters():
                if 'fc' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = False

        self.evaluation_model = model #
        self.datamodule.setup(stage = 'test',pretrain = False)
        self.test_loader = self.datamodule.test_dataloader(False)
        
        if self.evaluation_model.classification_type != 'multi-class':
            if self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'binary':
                self.ds_num_classes = 1
            elif self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'multi-label':
                self.ds_num_classes = 2
        else:
            self.ds_num_classes = self.datamodule.num_classes

        #fig = plot_metrics(self.train_losses, self.valid_losses, 'Loss')
        #fig = plot_metrics(self.train_accuracy, self.valid_accuracy, 'Accuracy')
        ## LOADING THE BEST MODEL
        self.evaluation_model.load_state_dict(torch.load(self.dsfilepath))
        test_loss, test_acc, preds, gts = self.valid_ds_epoch(self.evaluation_model, self.test_loader)
        
        # print(':::::Saving Predictions and One Hot Ground Truth data to \'.npy\' files:::::::')
        # np.save('test_set_preds.npy', preds)
        # np.save('test_set_onehot_gt.npy', gts)
        
        print(':::::::::::::::::::::::::::Performance Metrics:::::::::::::::::::::::::::::')
        if self.evaluation_model.classification_type != 'multi-class':
            test_perf_metrics = get_performance_metrics(gts, preds, [str(c) for c in list(range(self.ds_num_classes))])
            print(test_perf_metrics)
        else:
            print("\nTest Accuracy : {acc:.5f}, Test Loss : {loss:.5f}".format(acc = test_acc, loss = test_loss), flush = True)
        
        return {'_'.join([stage,mode,str(fracs).replace('.','p'),'test_loss']):test_loss,
                '_'.join([stage,mode,str(fracs).replace('.','p'),'test_acc']):test_acc}

    def train_ds_epoch(self, model, dataloader, optimizer):
        model.train()
        train_losses = 0
        train_accuracy = 0
        with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as tepoch:
            for step, batch in enumerate(tepoch):
                optimizer.zero_grad(set_to_none = True )
                train_loss, train_acc, _, _ = model.step(batch, step, False)
                #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                train_loss.backward()
                optimizer.step()
                train_losses += train_loss.item()
                train_accuracy += train_acc
                tepoch.set_postfix(loss = train_loss.item(), acc = train_accuracy/(step+1))
        train_losses = train_losses/(step+1)
        train_accuracy =train_accuracy/(step+1)
        return train_losses, train_accuracy

    def valid_ds_epoch(self, model, dataloader):
        model.eval()
        valid_losses = 0
        valid_accuracy = 0
        preds = np.array([]).reshape((0,self.ds_num_classes))
        gts = np.array([]).reshape((0,self.ds_num_classes))
        with torch.no_grad():
            with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as vepoch:
                for step, batch in enumerate(vepoch):
                    valid_loss, valid_acc, pred, gt = model.step(batch, step, False)
                    preds = np.append(preds, pred.numpy(), axis = 0)
                    if self.evaluation_model.classification_type == 'multi-class':
                        gt = torch.nn.functional.one_hot(gt, num_classes = self.ds_num_classes)
                    gts = np.append(gts, gt.numpy(), axis = 0)
                    #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                    valid_losses += valid_loss.item()
                    valid_accuracy += valid_acc
                    vepoch.set_postfix(loss = valid_loss.item(), acc = valid_accuracy/(step+1))
            valid_losses = valid_losses/(step+1)
            valid_accuracy = valid_accuracy/(step+1)
        return valid_losses, valid_accuracy, preds, gts

    def knn_eval(self,
                 dsmodel,
                 fracs: float = 1.0,
                 k: int = 200,
                 weights: str = 'distance',
                 algorithm: str = 'auto',
                 metric: str = 'minkowski',
                 net_model_path: str = None
                 ) -> Any:
        print(":::::::::::::::::K-NEAREST NEIGHBOUR CLASSIFICATION::::::::::::::::")
        if net_model_path is not None:
            dsmodel.load_state_dict(torch.load(net_model_path), strict = False)
        else:
            dsmodel.load_state_dict(torch.load(self.final_net_save_path), strict = False)

        knn_metric = {}
        for i in range(k, 201, 20):

            metrics = self.knn_downstream('knn_eval',
                                          dsmodel,
                                          fracs,
                                          i,
                                          weights,
                                          algorithm,
                                          metric
                                          )
            knn_metric = {**knn_metric, **metrics}
        return knn_metric

    def knn_cossim(self,
                 test_X,
                 train_X,
                 test_y,
                 train_y,
                 k,
                 temperature):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        test_X = torch.nn.functional.normalize(test_X, dim = -1)
        train_X = torch.nn.functional.normalize(train_X, dim = -1)
        sim_matrix = torch.mm(test_X, train_X.t())
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(train_y.expand(test_X.shape[0], -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / temperature).exp()

        # counts for each class
        one_hot_label = torch.zeros(test_X.shape[0] * k, self.ds_num_classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.to(torch.int64).view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(test_X.shape[0], -1, self.ds_num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)

        return pred_labels

    def knn_downstream(self,
                   stage: str,
                   model: nn.Module,
                   fracs: float = 1.0,
                   k: int = 200,
                   weights: str = 'distance',
                   algorithm: str = 'auto',
                   metric: str = 'minkowski'
                  ) -> None:

        stage = stage
        #model = model
        
        #for p in model.parameters():
        #    p.requires_grad = False
        #if not linear_eval:
        #    for p in model.net.base_encoder.parameters():
        #        p.requires_grad = True

        fracs = fracs
        self.knn_n_neighbors = k
        self.knn_weights = weights
        self.knn_algorithm = algorithm
        self.knn_metric = metric
        counter = 0

        self.evaluation_model = model #
        preds_shape = self.evaluation_model.base_encoder.fc.in_features
        #model.base_encoder.fc = nn.Identity()
        
        self.datamodule.setup(stage = 'train',pretrain = False, fracs = fracs)
        self.datamodule.setup(stage = 'valid',pretrain = False)
        self.train_loader = self.datamodule.train_dataloader(False)
        self.valid_loader = self.datamodule.valid_dataloader(False)
        self.datamodule.setup(stage = 'test',pretrain = False)
        self.test_loader = self.datamodule.test_dataloader(False)


        self.train_losses, self.valid_losses = np.array([]), np.array([])
        self.train_accuracy, self.valid_accuracy = np.array([]), np.array([])
        
        if self.evaluation_model.classification_type != 'multi-class':
            if self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'binary':
                self.ds_num_classes = 1
            elif self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'multi-label':
                self.ds_num_classes = 2
        else:
            self.ds_num_classes = self.datamodule.num_classes

        
        self.evaluation_model.eval()
        with torch.no_grad():
            tpreds = np.array([]).reshape((0,preds_shape))
            tgts = np.array([]) #.reshape((0,self.ds_num_classes))
            with tqdm(self.train_loader, unit = 'batch', total = len(self.train_loader)) as tepoch:
                for step, batch in enumerate(tepoch):
                    #x, y = batch #[b.cuda() for b in batch]
                    #z = self.evaluation_model.base_encoder(x.cuda()).cpu()
                    z, y = self.evaluation_model.step(batch, step, True)
                    tpreds = np.append(tpreds, z.numpy(), axis = 0)
                    #if self.evaluation_model.classification_type == 'multi-class':
                    #    y = torch.nn.functional.one_hot(y, num_classes = self.ds_num_classes)
                    tgts = np.append(tgts, y.numpy(), axis = 0)

            vpreds = np.array([]).reshape((0, preds_shape)) #self.ds_num_classes))
            vgts = np.array([])#.reshape((0,self.ds_num_classes))
            with tqdm(self.test_loader, unit = 'batch', total = len(self.test_loader)) as vepoch:
                for step, batch in enumerate(vepoch):
                    #x, y = batch #[b.cuda() for b in batch]
                    #z = self.evaluation_model.base_encoder(x.cuda()).cpu()
                    z, y = self.evaluation_model.step(batch, step, True)
                    vpreds = np.append(vpreds, z.numpy(), axis = 0)
                    #if self.evaluation_model.classification_type == 'multi-class':
                    #    y = torch.nn.functional.one_hot(y, num_classes = self.ds_num_classes)
                    vgts = np.append(vgts, y.numpy(), axis = 0)

        if self.knn_weights !='cosinesimilarity':

            knnclf = KNeighborsClassifier(n_neighbors = self.knn_n_neighbors, 
                                          weights = self.knn_weights,
                                          algorithm = self.knn_algorithm,
                                          metric = self.knn_metric)
            tgts = tgts[~np.isnan(tpreds)[:,0]]
            tpreds = tpreds[~np.isnan(tpreds)[:,0]]
            _ = knnclf.fit(tpreds, tgts)
            # print(vpreds)
            vgts = vgts[~np.isnan(vpreds)[:,0]]
            vpreds = vpreds[~np.isnan(vpreds)[:,0]]
            test_acc = knnclf.score(vpreds, vgts)

        else:
            pred_labels = self.knn_cossim(torch.from_numpy(vpreds), 
                                        torch.from_numpy(tpreds), 
                                        torch.from_numpy(vgts), 
                                        torch.from_numpy(tgts), 
                                        self.knn_n_neighbors, 0.5)
            test_acc = torch.sum((pred_labels[:, :1] == torch.from_numpy(vgts).unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_acc = test_acc/vgts.shape[0]
            #total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            #test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
            #                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

        self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'k']),k)
        
    
        # print(':::::Saving Predictions and Ground Truth data to \'.npy\' files:::::::')
        # np.save('knn_test_set_preds.npy', vpreds)
        # np.save('knn_test_set_onehot_gt.npy', vgts)
        
        print(':::::::::::::::::::::::::::::Performance Metrics:::::::::::::::::::::::::::::')
        if self.evaluation_model.classification_type != 'multi-class':
            test_perf_metrics = get_performance_metrics(gts, preds, [str(c) for c in list(range(self.ds_num_classes))])
            print(test_perf_metrics)
        else:
            print("\nFor k = {k:.1f} - Test Accuracy : {acc:.5f}".format(k = k, acc = test_acc, flush = True))
        
        return {'_'.join([stage,'k='+str(k),str(fracs).replace('.','p'),'test_acc']):test_acc}

    def extract_embeddings(self,
                           model,
                           model_path,
                           dataloader,
                           writer):
        if model_path is None:
            model.load_state_dict(torch.load(self.final_net_save_path), strict = False)
        else:
            model.load_state_dict(torch.load(model_path), strict = False)

        features = None
        labels = None
        labels_img = None
        with torch.no_grad():
            for x,y in dataloader:
                feats = model(x.cuda(non_blocking = True))
                if features == None:
                    features = feats.cpu().numpy()
                    labels = y
                    label_img = x.cpu().numpy().transpose(0,2,3,1)
                else:
                    features = np.append(features, feats.cpu().numpy(), axis = 0)
                    labels = np.append(labels, y, axis = 0)
                    labels_img = np.append(label_img, x.cpu().numpy().transpose(0,2,3,1), axis = 0)
        writer.add_embedding(features = features, metadata = labels, label_img = label_img)




