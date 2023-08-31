import os
#import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import pytorch_lightning as pl
sys.path.append('/home.guest/laposben/.local/lib/python3.10/site-packages/segmentation_models_pytorch')
sys.path.append('/home.guest/laposben/.local/lib/python3.10/site-packages')
sys.path.append('/local/temporary/segmentation_models_pytorch')
import segmentation_models_pytorch as smp
from skimage import io

from pprint import pprint
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, LearningRateMonitor


sys.path.append('/mnt/home.stud/laposben/Documents/Scripts/data')
from data_io import Camelyon16Dataset_openslide
from data_io import Camelyon16Dataset_openslide_rand, Camelyon16Dataset_openslide_negrand, Camelyon16Dataset_openslide_negrand_validset
import data_utils as utils

sys.path.append('/mnt/home.stud/laposben/Documents/Augmentation-PyTorch-Transforms-master')
import myTransforms
from NN import SegModel, MaxLossBatchCallback, SaveBestModel, LoadBestModel, SaveCheckPoint, UpdateDataClass

''''
choice of data augm, for now we can forget this step because it only gives poor results 
loading : 

'''


class Pipeline_Testing():

	def __init__(self, level, patch_size, name_dataset, valid_folder, gpu):
		self.level = level
		self.valid_folder = valid_folder
		self.patch_size = patch_size 
		self.name_dataset = name_dataset
		#self.path = os.path.join('/mnt/home.stud/laposben/Documents/Scripts/experiments', str(self.level), str(self.patch_size)) ##############################
		self.path = os.path.join('/datagrid/personal/laposben/Scripts/experiments', str(self.level), str(self.patch_size))
		self.gpu = gpu
	
	def get_loaders(self, batch_size, test = False):
	
		if not(test):
			loaders = []
			train_datasets = []
			self.batch_size = batch_size 
			n_cpu = os.cpu_count()
			for valid_folder in range(1,6):
				transform, transform_nomask = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.2), T.RandomVerticalFlip(0.2)]), T.Compose([T.ColorJitter(brightness = 0.2, contrast = 0.3, saturation = 0.3, hue = 0.04), T.GaussianBlur(kernel_size = 3, sigma= 0.8)])
				#train_dataset = Camelyon16Dataset_openslide_rand(level = self.level, tile_size = self.patch_size, name_dataset = self.name_dataset, folder_valid = valid_folder, valid = False, transform = transform, transform_nomask = transform_nomask) # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) https://towardsdatascience.com/data-augmentations-in-torchvision-5d56d70c372e  ########################################Put the rand
				train_dataset = Camelyon16Dataset_openslide_negrand_validset(level = self.level, patch_size = self.patch_size, name_dataset = self.name_dataset, folder_valid = valid_folder, valid = False, transform = transform, transform_nomask = transform_nomask)
				train_datasets.append(train_dataset)
				valid_dataset = Camelyon16Dataset_openslide(level = self.level, tile_size = self.patch_size, name_dataset = self.name_dataset, folder_valid = valid_folder, valid = True, transform = T.ToTensor()) #########no rand here
				train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=n_cpu, persistent_workers = False) ########works 0 and not n_cpu
				valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size,shuffle = False, num_workers=n_cpu, persistent_workers = False) #####used to have 4 as batch_size, think I did it because of GPU memory constraints 
				loaders.append([train_dataloader, valid_dataloader])
				
			return loaders, train_datasets, transform, transform_nomask
		
		else: #test training
			self.batch_size = batch_size 
			n_cpu = os.cpu_count()
			transform, transform_nomask = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.2), T.RandomVerticalFlip(0.2)]), T.Compose([T.ColorJitter(brightness = 0.2, contrast = 0.3, saturation = 0.3, hue = 0.04), T.GaussianBlur(kernel_size = 3, sigma= 0.8)])
			#train_dataset = Camelyon16Dataset_openslide_rand(level = self.level, tile_size = self.patch_size, name_dataset = self.name_dataset, folder_valid = -1, valid = False, test = True, transform = transform, transform_nomask = transform_nomask)
			train_dataset = Camelyon16Dataset_openslide_negrand(level = self.level, patch_size = self.patch_size, transform = transform, transform_nomask = transform_nomask)
			train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=n_cpu, persistent_workers = False)
			return train_dataloader, train_dataset, transform, transform_nomask
			
		
	def create_model(self, arch, encoder, **kwargs):
		
		#model = SegModel(arch, encoder, in_channels=3, out_classes=1, **kwargs) ################################## Dropout implemented so the resnet101 tried to not overfit 
		aux_params = dict(dropout=0.5)
		model = SegModel(arch, encoder, in_channels=3, out_classes=1, aux_params = aux_params, **kwargs) ################################# SegModelRAM
		return model 
		
	def write_results(self, name, batch_size, max_epochs, arch, encoder, name_model, val_folder):
        	
        	model = self.create_model(arch, encoder, name_model = name_model)
        	loaders = self.get_loaders(batch_size)
        	valid_dataloader = loaders[val_folder-1][1]
        	trainer_val = pl.Trainer(accelerator = 'gpu', devices= self.gpu, num_nodes=1)
        	valid_metrics = trainer_val.validate(model, dataloaders=valid_dataloader, verbose=False)
        	with open(os.path.join(self.path, 'description.txt'), 'a') as F:
        		line = f'name:{name};version:{val_folder-1};dataset:{self.name_dataset};folder_valid:{val_folder};arch:{arch};encoder:{encoder};epochs:{max_epochs};batch_size:{batch_size};gpus:1;results:{valid_metrics}\n'
        		F.write(line)
        	
	def main(self, name, batch_size, max_epochs, arch, encoder, name_model = None, description = None, precision = 32, name_weights = None):
		
        	loaders, train_datasets, transform, transform_nomask = self.get_loaders(batch_size)
        	gpu = self.gpu
        	name_model = f'name={name};ep={max_epochs};bs={batch_size};arch={arch};encoder={encoder};precision={precision}'
        	save_dir = os.path.join(self.path, 'logs')
        	for folder in range(5):
        		if os.path.exists(os.path.join(self.path, name_model + f'_val:{folder+1}.pt')):
        			continue
        		if self.valid_folder is None or folder == self.valid_folder - 1:
        			train_dataset = train_datasets[folder]
        			model = self.create_model(arch, encoder, save_path = self.path, name_model = name_model, valid_folder = folder + 1, train_dataset = train_dataset)
        			print('name_weights is not None', name_weights is not None)
        			if name_weights is not None:
        				model.load_state_dict(torch.load(os.path.join(self.path, name_weights + '.pt')))
        				print(f'model loaded with weights {name_weights}')
        			train_dataloader, valid_dataloader = loaders[folder]
        			utils.mkdir(os.path.join(save_dir, name))
        			trainer = pl.Trainer(accelerator = 'gpu', devices = gpu, max_epochs= max_epochs, accumulate_grad_batches = 1, check_val_every_n_epoch = 1, logger=pl.loggers.TensorBoardLogger(save_dir, name=name), precision = precision, callbacks=[SaveBestModel(), LoadBestModel(start_lr = 1e-4), SaveCheckPoint(), UpdateDataClass(), EarlyStopping(monitor='loss_valid_epoch', min_delta = 1e-5, patience = 41, verbose = True, mode = 'min'), LearningRateMonitor(logging_interval='epoch')])#, limit_train_batches=10, limit_val_batches=10) ############################changed the patience to 4. It was 3.
        			####### because I do an error because logs file was missing so I create it 
        			#if not os.path.exists(os.path.join(save_dir, name)):
        			#	os.mkdir(os.path.join(save_dir, name))
        			trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        			#state_dict = model.state_dict()
        			#torch.save(state_dict, os.path.join(self.path,name_model + f'_val:{folder+1}.pt'))      		
        			trainer_val = pl.Trainer(accelerator = 'gpu', devices = gpu, num_nodes=1,  callbacks=[SaveBestModel()])#, limit_val_batches=10) #permet de check val une derneire fois sur sur dernioere epoch il n y a pas eu de check val
        			valid_metrics = trainer_val.validate(model, dataloaders=valid_dataloader, verbose=False)	
        			with open(os.path.join(self.path, 'description.txt'), 'a') as F:
        				line = f'description={description}\nname:{name};version:{folder};dataset:{name_dataset};folder_valid:{folder+1};arch:{arch};encoder:{encoder};epochs:{max_epochs};batch_size:{batch_size};results:{valid_metrics};dataaugm:{transform}-{transform_nomask}\n'
        				F.write(line)
	
	def test(self, name, batch_size, max_epochs, arch, encoder, name_model = None, description = None, callbacks = True, precision = 32, name_weights = None):
        	
        	self.path = os.path.join(self.path, 'test')
        	train_dataloader, train_dataset, transform, transform_nomask = self.get_loaders(batch_size, test = True)	
        	gpu = self.gpu
        	name_model = f'name={name};ep={max_epochs};bs={batch_size};arch={arch};encoder={encoder};precision={precision}'
        	save_dir = os.path.join(self.path, 'logs')
        	model = self.create_model(arch, encoder, save_path = self.path, name_model = name_model, valid_folder = -1, train_dataset = train_dataset)
        	if name_weights is not None:
        		model.load_state_dict(torch.load(os.path.join(self.path, name_weights + '.pt')))
        	utils.mkdir(os.path.join(save_dir, name))
        	trainer = pl.Trainer(accelerator = 'gpu', devices = gpu, max_epochs= max_epochs, accumulate_grad_batches = 1, logger=pl.loggers.TensorBoardLogger(save_dir, name=name), precision = precision, callbacks=[SaveCheckPoint(), UpdateDataClass(), LearningRateMonitor(logging_interval='epoch')])#, limit_train_batches=15)
        	trainer.fit(model, train_dataloaders=train_dataloader)
        	
        	


if __name__ == "__main__":
	
	#level=0 gpu=0 batch_size=20 max_epochs=50 name_dataset='cut_[-1.0, 1.0]_numpos_86416_factor_1.0' valid_folder=4 name= python Pipeline_Testing.py 
	name_datasets = ['cut_[-1.0, 1.0]_numpos_86416_factor_1.0', 'cut_[0.65, 0.85]_numpos_24790_factor_2', 'cut_[0.6, 0.75]_numpos_7726_factor_2', 'cut_[0.6, 0.75]_numpos_2704_factor_2', 'cut_[0.4, 0.75]_numpos_1184_factor_2', 'cut_[0.15, 0.75]_numpos_611_factor_3']
	level, gpu, batch_size, max_epochs, name_dataset, valid_folder, name, patch_size = int(os.environ.get('level')), [int(os.environ.get('gpu'))], int(os.environ.get('batch_size')), int(os.environ.get('max_epochs')), os.environ.get('name_dataset'), os.environ.get('valid_folder'), os.environ.get('name'), os.environ.get('patch_size')
	if level and gpu and batch_size and max_epochs and name is not None:
		print(f"Level: {level}, gpu: {gpu}, batch_size: {batch_size}, max_epochs: {max_epochs}, name: {name}")
	else:
		print("value provided issue")
	
	if patch_size is None:
		patch_size = 512
	else:
		patch_size = int(patch_size)
	if name_dataset is None:
		name_datasets[level]
	print('name_dataset', name_dataset)
	if valid_folder is not None:
		valid_folder = int(valid_folder)
		print('valid folder: ', valid_folder)
	obj = Pipeline_Testing(level, patch_size, name_dataset, valid_folder, gpu)
	arch, encoder, name_weights, description = os.environ.get('arch'), os.environ.get('encoder'), os.environ.get('name_weights'), os.environ.get('description')
	if arch is None:
		arch = 'FPN'
	if encoder is None:
		encoder = 'resnet34'
	print('name_weights', name_weights)
	test_ = os.environ.get('test_')
	if test_ is None:
		obj.main(name, batch_size, max_epochs, arch, encoder, description = description, precision = 32, name_weights = name_weights)
	else:
		obj.test(name, batch_size, max_epochs, arch, encoder, description = description, precision = 32, name_weights = name_weights)
	
	#obj.write_results(name, batch_size, max_epochs, arch, encoder, name_model = f'ep=15;nodataaugm;bs=12;arch=FPN;encoder=resnet34;precision=32_val:{valid_folder}', val_folder=5)
	
	#test_=True level=0 gpu=1 batch_size=20 max_epochs=24 arch=UNET encoder=resnet34 name_dataset='cut_[-1.0, 1.0]_numpos_86416_factor_1.0' name=UNETcombinelossNegRand description='training on the whole dataset. No validdataset hence training blind. So: lr reduced every 8 epochs, model saved every 8 epochs. No early stopping. Using all the data augmentation. Using the data contained in the datasets 1 to 5. Using the negrand data class' python Pipeline_Testing.py 		
	#level=0 gpu=1 batch_size=20 max_epochs=50 arch=UNET encoder=resnet34 name_dataset='cut_[-1.0, 1.0]_numpos_86416_factor_1.0' name=UNETcombineloss description='combine loss + LoadBestModel Callback + patience 7 and 14 for reduce lr and stop' valid_folder=5 python Pipeline_Testing.py
	#level=5 gpu=0 batch_size=12 max_epochs=1 arch=FPN encoder=resnet34 name_dataset='cut_[0.2, 0.9]_numpos_611_factor_1.0' valid_folder=1 name=firsttestonduda python Pipeline_Testing.py	
		
	#level=2 gpu=2 batch_size=20 max_epochs=80 arch=UNET encoder=resnet34 name_dataset='cut_[-1.0, 1.0]_numpos_7851_factor_1.0' name=UNETcombinelossNegRand description='combine loss + LoadBestModel Callback + patience 20 and 41 for reduce lr ReduceOnPlateau and early stop. Training for 80 epochs, with the class neg rand to have random patches. Save every 20 epochs and load the best model, save best model also.' python Pipeline_Testing.py	
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			     	
        	
        
        
        
