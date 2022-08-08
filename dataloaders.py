import numpy as np
import pandas as pd
import shutil, os, requests, random, copy, sys
import collections, datetime
# import imageio
# from sklearn.transform import rotate, AffineTransform, warp, resize
# import skvideo.io as vidio
import torch
import torch.nn as nn
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from dicom_utils import load_dcmvolume

class SSLDataFrameDataset(Dataset):
	def __init__(self, phase, df, target_class, num_frames, transformations = None):
		super().__init__()
		self.phase = phase
		self.filenames_df = df
		self.target_class = target_class
		self.transforms = transformations
		self.num_frames = num_frames
		self.x1minframeidx = 0
		# self.mean = 0
		# self.std = 0
	# ===================================

	def load_volume(self, file_idx):
		filePoolLen = self.filenames_df.shape[0]
		file_idx = file_idx%filePoolLen #np.random.randint(0,filePoolLen)
		npy_file = np.load(self.filenames_df['filename'].iloc[file_idx])
		return npy_file
	
	# ===================================

	def get_frames(self, idx, repeat = False):
		image_volume = self.load_volume(idx)
		tot_frames = image_volume.shape[0]
		if not repeat:
			self.x1minframeidx = min(self.x1minframeidx, tot_frames - 16)
			frame_idxs = np.arange(self.x1minframeidx, self.x1minframeidx+16)
		else:
			frame_idxs = np.random.randint(max(0,self.x1minframeidx-tot_frames//10),tot_frames,size=self.num_frames)
		frames = np.array(image_volume[frame_idxs,:,:])
		self.x1minframeidx = np.min(frame_idxs)
		#print(frames.shape)
		return frames

	# ===================================
	
	def __len__(self):
		#return int(np.floor((len(self.filenames_df)/self.batch_size)))
		return len(self.filenames_df)

	# ====================================
	
	def __getitem__(self, idx):

		#GET CLIP FRAMES
		file_idx = idx #self.start_idx + bs
		
		if np.random.random() < 0.75:
			frame = self.get_frames(idx, False)
			
			original = torch.from_numpy(np.repeat(frame[:,None,:,:],3,axis=1).astype(np.float32))
			#print(original.shape)

			original = original / 255.0

			x1, x2 = self.augment(original)
		
		else:
			frame1 = self.get_frames(idx, False)
			frame2 = self.get_frames(idx, True if np.random.random() < 0.5 else False)

			original1 = torch.from_numpy(np.repeat(frame1[:,None,:,:],3,axis=1).astype(np.float32))
			#print(original.shape)
			original1 = original1 / 255.0

			original2 = torch.from_numpy(np.repeat(frame2[:,None,:,:],3,axis=1).astype(np.float32))
			#print(original.shape)
			original2 = original2 / 255.0

			x1, _ = self.augment(original1)
			x2, _ = self.augment(original2)

			# print(x1.mean(dim = (0,2,3), keepdim = True), x1.std(dim = (0,2,3), keepdim = True))

		x1 = (x1 - x1.mean(dim = (0,2,3), keepdim = True))/x1.std(dim = (0,2,3), keepdim = True)
		x2 = (x2 - x2.mean(dim = (0,2,3), keepdim = True))/x2.std(dim = (0,2,3), keepdim = True)

		x1 = x1.transpose(0,1).contiguous()
		x2 = x2.transpose(0,1).contiguous()

		y = self.filenames_df[self.target_class].values[idx]

		return x1.to(dtype = torch.float), x2.to(dtype = torch.float), y

	# ===================================     
	
	def on_epoch_end(self):
		self.filenames_df = self.filenames_df.sample(frac=1).reset_index(drop=True)

	# ==================================
 
	#applies randomly selected augmentations to each clip (same for each frame in the clip)
	def augment(self, x):
		if self.phase == 'train':
			x1, x2 = self.transforms(x)
		else:
			x1, x2 = self.transforms(x), self.transforms(x)
		return x1, x2

# ================================================================================

class SLDataFrameDataset(Dataset):
	def __init__(self, 
				 phase, 
				 df, 
				 class_name, 
				 num_frames, 
				 transformations = None, 
				 fracs = 1.0,
				 dsoversample = True,
				 dsbinary = True):
		super().__init__()
		self.phase = phase
		if self.phase != 'test':
			self.mult = 1
		else:
			self.mult = 5
		self.df = df
		self.fracs = fracs
		self.class_name = class_name
		self.num_frames = num_frames
		self.dsoversample = dsoversample
		self.dsbinary = dsbinary
		if self.fracs < 1.0:
			self.df, _ = train_test_split(self.df, test_size = 1 - fracs)
		# self.normalize = transforms.Normalize(mean, std)
		self.transforms = transformations

		self.global_idx = 0

		if self.phase == 'train':
			if self.dsbinary and self.dsoversample:
				num_1 = np.count_nonzero(self.df[self.class_name].values==1)
				num_0 = np.count_nonzero(self.df[self.class_name].values==0)
				if num_1 > num_0:
					print(f"Oversampling Negative Samples of {self.class_name}...")
					gap = num_1 - num_0
					df_zero = self.df[self.df[self.class_name].values == 0]
					self.df = self.df.append(df_zero.sample(n = gap, replace = True, random_state = 0), ignore_index = True)
				elif num_0 > num_1:
					print(f"Oversampling Positive Samples of {self.class_name}...")
					gap = num_0 - num_1
					df_one = self.df[self.df[self.class_name].values == 1]
					self.df = self.df.append(df_one.sample(n = gap, replace = True, random_state = 1), ignore_index = True)
				else:
					print(f"Class {self.class_name} is balanced")
				self.bin_pos_wts = torch.Tensor([1.0])
			if self.dsbinary and not self.dsoversample:
				num_1 = np.count_nonzero(self.df[self.class_name].values==1)
				num_0 = np.count_nonzero(self.df[self.class_name].values==0)
				self.bin_pos_wts = torch.Tensor([num_1/num_0])
			if not self.dsbinary and not self.oversample:
				self.bin_pos_wts = np.array([])
				for c in self.class_name:
					num_1 = np.count_nonzero(self.df[c].values==1)
					num_0 = np.count_nonzero(self.df[c].values==0)
					self.bin_pos_wts = np.append(self.bin_pos_wts, num_1/num_0)
				self.bin_pos_wts = torch.from_numpy(self.bin_pos_wts)

	# ===================================
	def load_volume(self, file_idx):
		filePoolLen = self.df.shape[0]
		file_idx = file_idx%filePoolLen #np.random.randint(0,filePoolLen)
		npy_file = np.load(self.df['filename'].iloc[file_idx])
		return npy_file
	
	# ===================================

	def get_frames(self, idx):
		image_volume = self.load_volume(idx)
		tot_frames = image_volume.shape[0]
		# frame_idxs = np.random.randint(0,tot_frames,size=self.num_frames)
		frame_idxs = np.array(sorted(random.sample(list(range(tot_frames)),min(tot_frames,16))))
		frames = np.array(image_volume[frame_idxs,:,:])
		#print(frames.shape)
		return frames

	# ===================================
	
	def __getitem__(self, idx):

		#GET CLIP FRAMES
		if self.global_idx == self.__len__():
			self.global_idx = 0
			
		file_idx = self.global_idx//self.mult #self.start_idx + bs

		
		frame = self.get_frames(file_idx)
		
		original = torch.from_numpy(np.repeat(frame[:,None,:,:],3,axis=1).astype(np.float32))
		#print(original.shape)

		original = original / 255.0

		x1 = self.augment(original)
		# plt.imshow(x1[0].numpy().transpose(1,2,0))
		# plt.show()
		m = x1.mean(dim = (0,2,3), keepdim = True)
		s = x1.std(dim = (0,2,3), keepdim = True)

		x1 = (x1 - m)/s
		# x2 = (x2 - x2.mean(dim = 0))/x2.std(dim = 0)
		x1 = x1.transpose(0,1).contiguous()

		y = self.df[self.class_name].values[self.global_idx//self.mult]

		self.global_idx += 1

		return x1.to(dtype = torch.float), y
	# ===================================
		
	def __len__(self):
		return len(self.df)*self.mult

	# ===================================
	# ===================================

	#shuffles the dataset at the end of each epoch
	def on_epoch_end(self):
		self.global_idx = 0
		self.df = self.df.sample(frac=1).reset_index(drop=True)

	# ===================================

	#applies randomly selected augmentations to each clip (same for each frame in the clip)
	def augment(self, x):
		if self.transforms is not None:
			x = self.transforms(x)
		# x = self.normalize(x)
		return x

# ================================================================================

class SSLDCMDFDataset(Dataset):
    def __init__(self, phase, df, target_class, num_frames, transformations = None):
        super().__init__()
        self.phase = phase
        self.filenames_df = df
        self.target_class = target_class
        self.transforms = transformations
        self.num_frames = num_frames
        self.x1minframeidx = 0
        # self.mean = 0
        # self.std = 0
        # ===================================
        self.load_volume = load_dcmvolume

    # def load_volume(self, file_idx):
    #    filePoolLen = self.filenames_df.shape[0]
    #    file_idx = file_idx%filePoolLen #np.random.randint(0,filePoolLen)
    #    npy_file = np.load(self.filenames_df['filename'].iloc[file_idx])
    #    return npy_file

        # ===================================

    def get_frames(self, idx, repeat = False):
    	filepath = self.filenames_df['filepath'].iloc[idx]
        image_volume = self.load_volume(filepath)
        tot_frames = image_volume.shape[0]
        if not repeat:
            self.x1minframeidx = min(self.x1minframeidx, tot_frames - 16)
            frame_idxs = np.arange(self.x1minframeidx, self.x1minframeidx+16)
        else:
            frame_idxs = np.random.randint(max(0,self.x1minframeidx-tot_frames//10),tot_frames,size=self.num_frames)
        frames = image_volume[frame_idxs,:,:]
        self.x1minframeidx = np.min(frame_idxs)
        #print(frames.shape)
        return frames

        # ===================================

    def __len__(self):
        #return int(np.floor((len(self.filenames_df)/self.batch_size)))
        return len(self.filenames_df)

        # ====================================

    def __getitem__(self, idx):

        #GET CLIP FRAMES
        file_idx = idx #self.start_idx + bs

        if np.random.random() < 0.75:
            frame = self.get_frames(idx, False)

            original = frame.unsqueeze(1).repeat(1,3,1,1)
            #print(original.shape)
            #original = original / 255.0

            x1, x2 = self.augment(original)

        else:
            frame1 = self.get_frames(idx, False)
            frame2 = self.get_frames(idx, True if np.random.random() < 0.5 else False)

            original1 = frame1.unsqueeze(1).repeat(1,3,1,1)
            #torch.from_numpy(np.repeat(frame1[:,None,:,:],3,axis=1).astype(np.float32))
            #print(original.shape)
            #original1 = original1 / 255.0

            original2 = frame2.unsqueeze(1).repeat(1,3,1,1)
            #torch.from_numpy(np.repeat(frame2[:,None,:,:],3,axis=1).astype(np.float32))
            #print(original.shape)
            #original2 = original2 / 255.0

            x1, _ = self.augment(original1)
            x2, _ = self.augment(original2)

            # print(x1.mean(dim = (0,2,3), keepdim = True), x1.std(dim = (0,2,3), keepdim = True))

        x1 = (x1 - x1.mean(dim = (0,2,3), keepdim = True))/x1.std(dim = (0,2,3), keepdim = True)
        x2 = (x2 - x2.mean(dim = (0,2,3), keepdim = True))/x2.std(dim = (0,2,3), keepdim = True)

        x1 = x1.transpose(0,1).contiguous()
        x2 = x2.transpose(0,1).contiguous()

        y = self.filenames_df[self.target_class].values[idx]

        return x1.to(dtype = torch.float), x2.to(dtype = torch.float), y

        # ===================================     

    def on_epoch_end(self):
        self.filenames_df = self.filenames_df.sample(frac=1).reset_index(drop=True)

        # ==================================

    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, x):
        if self.phase == 'train':
            x1, x2 = self.transforms(x)
        else:
            x1, x2 = self.transforms(x), self.transforms(x)
        return x1, x2

        # ================================================================================

class SLDCMDFDataset(Dataset):
    def __init__(self, 
         phase, 
         df, 
         class_name, 
         num_frames, 
         transformations = None, 
         fracs = 1.0,
         dsoversample = True,
         dsbinary = True):
        super().__init__()
        self.phase = phase
        if self.phase != 'test':
            self.mult = 1
        else:
            self.mult = 5
        self.df = df
        self.fracs = fracs
        self.class_name = class_name
        self.num_frames = num_frames
        self.dsoversample = dsoversample
        self.dsbinary = dsbinary
        if self.fracs < 1.0:
            self.df, _ = train_test_split(self.df, test_size = 1 - fracs)
        # self.normalize = transforms.Normalize(mean, std)
        self.transforms = transformations
        
        self.load_volume = load_dcmvolume

        self.global_idx = 0

        if self.phase == 'train':
            if self.dsbinary and self.dsoversample:
                num_1 = np.count_nonzero(self.df[self.class_name].values==1)
                num_0 = np.count_nonzero(self.df[self.class_name].values==0)
                if num_1 > num_0:
                    print(f"Oversampling Negative Samples of {self.class_name}...")
                    gap = num_1 - num_0
                    df_zero = self.df[self.df[self.class_name].values == 0]
                    self.df = self.df.append(df_zero.sample(n = gap, replace = True, random_state = 0), ignore_index = True)
                elif num_0 > num_1:
                    print(f"Oversampling Positive Samples of {self.class_name}...")
                    gap = num_0 - num_1
                    df_one = self.df[self.df[self.class_name].values == 1]
                    self.df = self.df.append(df_one.sample(n = gap, replace = True, random_state = 1), ignore_index = True)
                else:
                    print(f"Class {self.class_name} is balanced")
                self.bin_pos_wts = torch.Tensor([1.0])
            if self.dsbinary and not self.dsoversample:
                num_1 = np.count_nonzero(self.df[self.class_name].values==1)
                num_0 = np.count_nonzero(self.df[self.class_name].values==0)
                self.bin_pos_wts = torch.Tensor([num_1/num_0])
            if not self.dsbinary and not self.oversample:
                self.bin_pos_wts = np.array([])
                for c in self.class_name:
                    num_1 = np.count_nonzero(self.df[c].values==1)
                    num_0 = np.count_nonzero(self.df[c].values==0)
                    self.bin_pos_wts = np.append(self.bin_pos_wts, num_1/num_0)
                self.bin_pos_wts = torch.from_numpy(self.bin_pos_wts)

    # ===================================
    '''def load_volume(self, file_idx):
        filePoolLen = self.df.shape[0]
        file_idx = file_idx%filePoolLen #np.random.randint(0,filePoolLen)
        npy_file = np.load(self.df['filename'].iloc[file_idx])
        return npy_file'''

    # ===================================

    def get_frames(self, idx):
    	filepath = self.filenames_df['filepath'].iloc[idx]
        image_volume = self.load_volume(filepath)
        tot_frames = image_volume.shape[0]
        # frame_idxs = np.random.randint(0,tot_frames,size=self.num_frames)
        frame_idxs = np.array(sorted(random.sample(list(range(tot_frames)),min(tot_frames,16))))
        frames = np.array(image_volume[frame_idxs,:,:])
        #print(frames.shape)
        return frames

    # ===================================

    def __getitem__(self, idx):

        #GET CLIP FRAMES
        if self.global_idx == self.__len__():
            self.global_idx = 0

        file_idx = self.global_idx//self.mult #self.start_idx + bs
        filepath = self.filenames_df['filepath'].iloc[file_idx]

        frame = self.get_frames(filepath)

        original = frame.unsqueeze(1).repeat(1,3,1,1)
        # torch.from_numpy(np.repeat(frame[:,None,:,:],3,axis=1).astype(np.float32))
        #print(original.shape)
        #original = original / 255.0

        x1 = self.augment(original)
        # plt.imshow(x1[0].numpy().transpose(1,2,0))
        # plt.show()
        m = x1.mean(dim = (0,2,3), keepdim = True)
        s = x1.std(dim = (0,2,3), keepdim = True)

        x1 = (x1 - m)/s
        # x2 = (x2 - x2.mean(dim = 0))/x2.std(dim = 0)
        x1 = x1.transpose(0,1).contiguous()

        y = self.df[self.class_name].values[self.global_idx//self.mult]

        self.global_idx += 1

        return x1.to(dtype = torch.float), y
    # ===================================

    def __len__(self):
        return len(self.df)*self.mult

    # ===================================
    # ===================================

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.global_idx = 0
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    # ===================================

    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        # x = self.normalize(x)
        return x

# ================================================================================

class MRNetDataModule(nn.Module):
	def __init__(self, 
				 path : str, 
				 class_name : str = 'acl', 
				 plane : str = 'sagittal',
				 pt_num_frames : int = 4,
				 ds_num_frames : int = 16,
				 batch_size : int = 32,
				 ds_batch_size : int = 1,
				 transforms : nn.Module = None,
				 data_dim : int = 224,
				 dsoversample : bool = True,
				 dsbinary: bool = True,
				 **kwargs: Any) -> None :
		self.mrnet_path = path #'E:/Siladittya_JRF/Dataset/MRNet-v1.0'
		self.class_name = class_name
		self.plane = plane
		self.classes = ['abn', 'acl', 'men']
		self.planes = ['sagittal', 'coronal', 'axial']

		self.train_dir = os.path.join(self.mrnet_path, 'train')
		self.valid_dir = os.path.join(self.mrnet_path, 'valid')
		self.base_dir = self.mrnet_path

		self.pt_num_frames = pt_num_frames
		self.ds_num_frames = ds_num_frames
		self.batch_size = batch_size
		self.ds_batch_size = ds_batch_size
		self.data_dim = data_dim
		self.transforms = transforms
		self.dsoversample = dsoversample
		self.dsbinary = dsbinary

		if isinstance(self.class_name, list): 
			if len(self.class_name) > 1:
				self.num_classes = len(self.class_name)
				self.classification_type = 'multi-label'
			if len(self.class_name) == 1:
				self.num_classes = 2	
				self.classification_type = 'binary'
		elif isinstance(self.class_name, str):
			self.num_classes = 2
			self.classification_type = 'binary'

		# ==================================================

		self.trabn = pd.read_csv(os.path.join(self.mrnet_path,'train-abnormal.csv'), header = None)
		self.tracl = pd.read_csv(os.path.join(self.mrnet_path,'train-acl.csv'), header = None)
		self.trmen = pd.read_csv(os.path.join(self.mrnet_path,'train-meniscus.csv'), header = None)

		self.trabn.columns = ['patient_id', 'label']
		self.tracl.columns = ['patient_id', 'label']
		self.trmen.columns = ['patient_id', 'label']

		self.tr_multilabel = self.trabn.merge(self.tracl, on = 'patient_id').merge(self.trmen, on = 'patient_id')
		self.tr_multilabel.columns = ['patient_id', 'abn', 'acl', 'men']

		# ===================================================

		self.testabn = pd.read_csv(os.path.join(self.mrnet_path,'valid-abnormal.csv'), header = None)
		self.testacl = pd.read_csv(os.path.join(self.mrnet_path,'valid-acl.csv'), header = None)
		self.testmen = pd.read_csv(os.path.join(self.mrnet_path,'valid-meniscus.csv'), header = None)

		self.testabn.columns = ['patient_id', 'label']
		self.testacl.columns = ['patient_id', 'label']
		self.testmen.columns = ['patient_id', 'label']

		self.test_multilabel = self.testabn.merge(self.testacl, on = 'patient_id').merge(self.testmen, on = 'patient_id')
		self.test_multilabel.columns = ['patient_id', 'abn', 'acl', 'men']

		# ===================================================
		# ===================================================

		# for c in self.planes:
		self.tr_filenames_df = pd.DataFrame(columns=['patient_id','filename'])
		self.tr_filenames_df['filename'] = os.listdir(os.path.join(self.mrnet_path,'train',self.plane))

		self.tr_filenames_df['patient_id'] = self.tr_filenames_df.apply(lambda x : int(x['filename'][:-4]),axis=1)
		self.tr_filenames_df['filename'] = self.tr_filenames_df.apply(lambda x : os.path.join(self.mrnet_path,'train',self.plane,x['filename']),axis=1)

		self.tr_filenames_df = self.tr_filenames_df[list(('patient_id','filename'))]
		self.tr_filenames_df.sort_values(by=['patient_id'],ascending=True,inplace=True,ignore_index=True)
		self.tr_df = self.tr_filenames_df.merge(self.tr_multilabel, on = 'patient_id')

		self.tr_df, self.val_df = train_test_split(self.tr_df, test_size = 0.1)

		# print(np.count_nonzero(self.val_df['acl'].values))

		# ===================================================
		
		self.test_filenames_df = pd.DataFrame(columns=['patient_id','filename'])
		self.test_filenames_df['filename'] = os.listdir(os.path.join(self.mrnet_path,'valid',self.plane))

		self.test_filenames_df['patient_id'] = self.test_filenames_df.apply(lambda x : int(x['filename'][:-4]),axis=1)
		self.test_filenames_df['filename'] = self.test_filenames_df.apply(lambda x : os.path.join(self.mrnet_path,'valid',self.plane,x['filename']),axis=1)

		self.test_filenames_df = self.test_filenames_df[list(('patient_id','filename'))]
		self.test_filenames_df.sort_values(by=['patient_id'],ascending=True,inplace=True,ignore_index=True)
		self.test_df = self.test_filenames_df.merge(self.test_multilabel, on = 'patient_id')
		# ===================================================

	def setup(self, stage: Optional[str] = None, pretrain : Optional[bool] = True, fracs: Optional[float] = 1.0):
		#transforms
		if stage == 'train':
			self.train_transforms = self.transforms
			if pretrain:
				self.traingen = SSLDataFrameDataset('train', self.tr_df, self.class_name, self.pt_num_frames, self.train_transforms)
			else:
				self.traingen = SLDataFrameDataset('train', 
													self.tr_df, 
													self.class_name, 
													self.ds_num_frames, 
													transforms.RandomResizedCrop(self.data_dim,(0.8,1.0)), 
													fracs,
													self.dsoversample,
													self.dsbinary)

		if stage == 'valid':
			self.valid_transforms = transforms.Compose([transforms.RandomResizedCrop(self.data_dim,(0.8,1.0))])
			# #None 
			#transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
			if pretrain:
				self.validgen = SSLDataFrameDataset('valid', self.val_df, self.class_name, self.pt_num_frames, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))
			else:
				self.validgen = SLDataFrameDataset('valid', self.val_df, self.class_name, self.ds_num_frames, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))

		if stage == 'test':
			self.test_transforms = transforms.Compose([transforms.RandomResizedCrop(self.data_dim,(0.8,1.0))])
			#None 
			#transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
			if pretrain:
				self.testgen = SSLDataFrameDataset('test', self.test_df, self.class_name, self.ds_num_frames, self.test_transforms)
			else:
				self.testgen = SLDataFrameDataset('test', self.test_df, self.class_name, self.ds_num_frames, self.test_transforms)

	def train_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			trainloader = DataLoader(self.traingen, batch_size = self.batch_size, shuffle = True, drop_last = True)
		else:
			trainloader = DataLoader(self.traingen, batch_size = self.ds_batch_size, shuffle = True, drop_last = True)
		return trainloader

	def valid_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			validloader = DataLoader(self.validgen, batch_size = self.ds_batch_size, drop_last = True)
		else:
			validloader = DataLoader(self.validgen, batch_size = self.ds_batch_size, drop_last = True)
		return validloader

	def test_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			testloader = DataLoader(self.testgen, batch_size = self.ds_batch_size, drop_last = True)
		else:
			testloader = DataLoader(self.testgen, batch_size = self.ds_batch_size, drop_last = True)
		return testloader

class RSNABraTSModule(nn.Module):
	def __init__(self, 
				 path : str, 
				 class_name : str = 'MGMT_value', 
				 plane : str = 'T1w',
				 pt_num_frames : int = 16,
				 ds_num_frames : int = 16,
				 batch_size : int = 4,
				 ds_batch_size : int = 4,
				 transforms : nn.Module = None,
				 data_dim : int = 224,
				 dsoversample : bool = True,
				 dsbinary: bool = True,
				 **kwargs: Any) -> None :
		self.rsna_path = path #'E:/Siladittya_JRF/Dataset/MRNet-v1.0'
		self.class_name = class_name
		self.plane = plane
		self.classes = ['1', '0']
		self.planes = ['FLAIR', 'T1w', 'T1wCE','T2w']

		self.train_dir = os.path.join(self.rsna_path, 'train')
		self.test_dir = os.path.join(self.rsna_path, 'test')
		self.base_dir = self.rsna_path

		self.exclude_ids = ['00109', '00123', '00709']

		self.pt_num_frames = pt_num_frames
		self.ds_num_frames = ds_num_frames
		self.batch_size = batch_size
		self.ds_batch_size = ds_batch_size
		self.data_dim = data_dim
		self.transforms = transforms
		self.dsoversample = dsoversample
		self.dsbinary = dsbinary

		# if isinstance(self.class_name, list): 
		# 	if len(self.class_name) > 1:
		# 		self.num_classes = len(self.class_name)
		# 		self.classification_type = 'multi-label'
		# 	if len(self.class_name) == 1:
		# 		self.num_classes = 2	
		# 		self.classification_type = 'binary'
		# elif isinstance(self.class_name, str):
		self.num_classes = 2
		self.classification_type = 'binary'

		self.brats21_ids = os.listdir(self.train_dir)

		self.train_df = pd.read_csv(os.path.join(self.base_dir, 'train_labels.csv'))
		self.train_df['filepath'] = self.train_df['BraTS21ID'].apply(lambda x:os.path.join(self.train_dir, str(x).zfill(5), self.plane))
		self.train_df = self.train_df[~self.train_df['BraTS21ID'].isin(list(map(int,self.exclude_ids)))]
		self.test_df = pd.DataFrame(columns = ['BraTS21ID', 'MGMT_value'])
		self.test_df['BraTS21ID'] = list(map(int, os.listdir(self.test_dir)))

		# print(self.train_df)
		# print(self.test_df)
		# =================================================
		self.train_df, self.valid_df = train_test_split(self.train_df, test_size = 0.1)
		# ===================================================
		# ===================================================

	def setup(self, stage: Optional[str] = None, pretrain : Optional[bool] = True, fracs: Optional[float] = 1.0):
		#transforms
		if stage == 'train':
			self.train_transforms = self.transforms
			if pretrain:
				self.traingen = SSLDataFrameDataset('train', self.train_df, self.class_name, self.pt_num_frames, self.train_transforms)
			else:
				self.traingen = SLDataFrameDataset('train', 
													self.train_df, 
													self.class_name, 
													self.ds_num_frames, 
													transforms.RandomResizedCrop(self.data_dim,(0.8,1.0)), 
													fracs,
													self.dsoversample,
													self.dsbinary)

		if stage == 'valid':
			self.valid_transforms = transforms.Compose([transforms.RandomResizedCrop(self.data_dim,(0.8,1.0))])
			# #None 
			#transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
			if pretrain:
				self.validgen = SSLDataFrameDataset('valid', self.valid_df, self.class_name, self.pt_num_frames, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))
			else:
				self.validgen = SLDataFrameDataset('valid', self.valid_df, self.class_name, self.ds_num_frames, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))

		if stage == 'test':
			self.test_transforms = transforms.Compose([transforms.RandomResizedCrop(self.data_dim,(0.8,1.0))])
			#None 
			#transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
			if pretrain:
				self.testgen = SSLDataFrameDataset('test', self.test_df, self.class_name, self.ds_num_frames, self.test_transforms)
			else:
				self.testgen = SLDataFrameDataset('test', self.test_df, self.class_name, self.ds_num_frames, self.test_transforms)

	def train_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			trainloader = DataLoader(self.traingen, batch_size = self.batch_size, shuffle = True, drop_last = True)
		else:
			trainloader = DataLoader(self.traingen, batch_size = self.ds_batch_size, shuffle = True, drop_last = True)
		return trainloader

	def valid_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			validloader = DataLoader(self.validgen, batch_size = self.ds_batch_size, drop_last = True)
		else:
			validloader = DataLoader(self.validgen, batch_size = self.ds_batch_size, drop_last = True)
		return validloader

	def test_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			testloader = DataLoader(self.testgen, batch_size = self.ds_batch_size, drop_last = True)
		else:
			testloader = DataLoader(self.testgen, batch_size = self.ds_batch_size, drop_last = True)
		return testloader

class ECGArythmiaDataModule(nn.Module):
	def __init__(self, 
				 path : str, 
				 class_name : str = 'acl', 
				 plane : str = 'sagittal',
				 pt_num_frames : int = 4,
				 ds_num_frames : int = 16,
				 batch_size : int = 32,
				 ds_batch_size : int = 1,
				 transforms : nn.Module = None,
				 data_dim : int = 224,
				 dsoversample : bool = True,
				 dsbinary: bool = True,
				 **kwargs: Any) -> None :
		self.mrnet_path = path #'E:/Siladittya_JRF/Dataset/MRNet-v1.0'
		self.class_name = class_name
		self.plane = plane
		self.classes = ['abn', 'acl', 'men']
		self.planes = ['sagittal', 'coronal', 'axial']

		self.train_dir = os.path.join(self.mrnet_path, 'train')
		self.valid_dir = os.path.join(self.mrnet_path, 'valid')
		self.base_dir = self.mrnet_path

		self.pt_num_frames = pt_num_frames
		self.ds_num_frames = ds_num_frames
		self.batch_size = batch_size
		self.ds_batch_size = ds_batch_size
		self.data_dim = data_dim
		self.transforms = transforms
		self.dsoversample = dsoversample
		self.dsbinary = dsbinary

		if isinstance(self.class_name, list): 
			if len(self.class_name) > 1:
				self.num_classes = len(self.class_name)
				self.classification_type = 'multi-label'
			if len(self.class_name) == 1:
				self.num_classes = 2	
				self.classification_type = 'binary'
		elif isinstance(self.class_name, str):
			self.num_classes = 2
			self.classification_type = 'binary'

		self.oversample = True
		# ==================================================

		self.trabn = pd.read_csv(os.path.join(self.mrnet_path,'train-abnormal.csv'), header = None)
		self.tracl = pd.read_csv(os.path.join(self.mrnet_path,'train-acl.csv'), header = None)
		self.trmen = pd.read_csv(os.path.join(self.mrnet_path,'train-meniscus.csv'), header = None)

		self.trabn.columns = ['patient_id', 'label']
		self.tracl.columns = ['patient_id', 'label']
		self.trmen.columns = ['patient_id', 'label']

		self.tr_multilabel = self.trabn.merge(self.tracl, on = 'patient_id').merge(self.trmen, on = 'patient_id')
		self.tr_multilabel.columns = ['patient_id', 'abn', 'acl', 'men']

		# ===================================================

		self.testabn = pd.read_csv(os.path.join(self.mrnet_path,'valid-abnormal.csv'), header = None)
		self.testacl = pd.read_csv(os.path.join(self.mrnet_path,'valid-acl.csv'), header = None)
		self.testmen = pd.read_csv(os.path.join(self.mrnet_path,'valid-meniscus.csv'), header = None)

		self.testabn.columns = ['patient_id', 'label']
		self.testacl.columns = ['patient_id', 'label']
		self.testmen.columns = ['patient_id', 'label']

		self.test_multilabel = self.testabn.merge(self.testacl, on = 'patient_id').merge(self.testmen, on = 'patient_id')
		self.test_multilabel.columns = ['patient_id', 'abn', 'acl', 'men']

		# ===================================================
		# ===================================================

		# for c in self.planes:
		self.tr_filenames_df = pd.DataFrame(columns=['patient_id','filename'])
		self.tr_filenames_df['filename'] = os.listdir(os.path.join(self.mrnet_path,'train',self.plane))

		self.tr_filenames_df['patient_id'] = self.tr_filenames_df.apply(lambda x : int(x['filename'][:-4]),axis=1)
		self.tr_filenames_df['filename'] = self.tr_filenames_df.apply(lambda x : os.path.join(self.mrnet_path,'train',self.plane,x['filename']),axis=1)

		self.tr_filenames_df = self.tr_filenames_df[list(('patient_id','filename'))]
		self.tr_filenames_df.sort_values(by=['patient_id'],ascending=True,inplace=True,ignore_index=True)
		self.tr_df = self.tr_filenames_df.merge(self.tr_multilabel, on = 'patient_id')

		self.tr_df, self.val_df = train_test_split(self.tr_df, test_size = 0.1)

		# print(np.count_nonzero(self.val_df['acl'].values))

		# ===================================================
		
		self.test_filenames_df = pd.DataFrame(columns=['patient_id','filename'])
		self.test_filenames_df['filename'] = os.listdir(os.path.join(self.mrnet_path,'valid',self.plane))

		self.test_filenames_df['patient_id'] = self.test_filenames_df.apply(lambda x : int(x['filename'][:-4]),axis=1)
		self.test_filenames_df['filename'] = self.test_filenames_df.apply(lambda x : os.path.join(self.mrnet_path,'valid',self.plane,x['filename']),axis=1)

		self.test_filenames_df = self.test_filenames_df[list(('patient_id','filename'))]
		self.test_filenames_df.sort_values(by=['patient_id'],ascending=True,inplace=True,ignore_index=True)
		self.test_df = self.test_filenames_df.merge(self.test_multilabel, on = 'patient_id')
		# ===================================================

	def setup(self, stage: Optional[str] = None, pretrain : Optional[bool] = True, fracs: Optional[float] = 1.0):
		#transforms
		if stage == 'train':
			self.train_transforms = self.transforms
			if pretrain:
				self.traingen = SSLDataFrameDataset('train', self.tr_df, self.class_name, self.pt_num_frames, self.train_transforms)
			else:
				self.traingen = SLDataFrameDataset('train', 
													self.tr_df, 
													self.class_name, 
													self.ds_num_frames, 
													transforms.RandomResizedCrop(self.data_dim,(0.8,1.0)), 
													fracs,
													self.dsoversample,
													self.dsbinary)

		if stage == 'valid':
			self.valid_transforms = transforms.Compose([transforms.RandomResizedCrop(self.data_dim,(0.8,1.0))])
			# #None 
			#transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
			if pretrain:
				self.validgen = SSLDataFrameDataset('valid', self.val_df, self.class_name, self.pt_num_frames, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))
			else:
				self.validgen = SLDataFrameDataset('valid', self.val_df, self.class_name, self.ds_num_frames, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))

		if stage == 'test':
			self.test_transforms = transforms.Compose([transforms.RandomResizedCrop(self.data_dim,(0.8,1.0))])
			#None 
			#transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
			if pretrain:
				self.testgen = SSLDataFrameDataset('test', self.test_df, self.class_name, self.ds_num_frames, self.test_transforms)
			else:
				self.testgen = SLDataFrameDataset('test', self.test_df, self.class_name, self.ds_num_frames, self.test_transforms)

	def train_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			trainloader = DataLoader(self.traingen, batch_size = self.batch_size, shuffle = True, drop_last = True)
		else:
			trainloader = DataLoader(self.traingen, batch_size = self.ds_batch_size, shuffle = True, drop_last = True)
		return trainloader

	def valid_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			validloader = DataLoader(self.validgen, batch_size = self.ds_batch_size, drop_last = True)
		else:
			validloader = DataLoader(self.validgen, batch_size = self.ds_batch_size, drop_last = True)
		return validloader

	def test_dataloader(self, pretrain : Optional[bool] = True):
		if pretrain:
			testloader = DataLoader(self.testgen, batch_size = self.ds_batch_size, drop_last = True)
		else:
			testloader = DataLoader(self.testgen, batch_size = self.ds_batch_size, drop_last = True)
		return testloader




