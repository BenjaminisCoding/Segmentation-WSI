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
from data_io import Camelyon16Dataset_openslide_rand
import data_utils as utils

sys.path.append('/mnt/home.stud/laposben/Documents/Augmentation-PyTorch-Transforms-master')
import myTransforms
from NN import SegModel, MaxLossBatchCallback, SaveBestModel, LoadBestModel


def create_model(arch, encoder, **kwargs):
	
	aux_params = dict(dropout=0.5)
	model = SegModel(arch, encoder, in_channels=3, out_classes=1, aux_params = aux_params, **kwargs) 
	return model 

def get_valid_dataset(level, patch_size, name_dataset, valid_folder, batch_size):

	n_cpu = os.cpu_count()
	valid_dataset = Camelyon16Dataset_openslide(level = level, tile_size = patch_size, name_dataset = name_dataset, folder_valid = valid_folder, valid = True, transform = T.ToTensor())
	valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size,shuffle = False, num_workers=n_cpu, persistent_workers = False)
	return valid_dataloader 
	
def main(level, patch_size, name_dataset, valid_folder, batch_size, arch, encoder, name_weights, gpu):

	valid_dataloader = get_valid_dataset(level, patch_size, name_dataset, valid_folder, batch_size)
	model = create_model(arch, encoder, save_path = '.', name_model = '.', valid_folder = 0)
	model.load_state_dict(torch.load(os.path.join(os.path.join('/datagrid/personal/laposben/Scripts/experiments', str(level), str(patch_size)), name_weights)))
	trainer_val = pl.Trainer(accelerator = 'gpu', devices = gpu)#, limit_val_batches=10)
	valid_metrics = trainer_val.validate(model, dataloaders=valid_dataloader, verbose=False)
	print(valid_metrics )
	return valid_metrics 
        
if __name__ == '__main__':

	#batch_size=30 level=0 gpu=0 name_dataset='cut_[-1.0, 1.0]_numpos_86416_factor_1.0' arch=UNET encoder=resnet34 valid_folder=1 name_weights= python see_results.py
	level, gpu, batch_size, name_dataset, valid_folder, patch_size, arch, encoder, name_weights = int(os.environ.get('level')), os.environ.get('gpu'), int(os.environ.get('batch_size')), os.environ.get('name_dataset'), os.environ.get('valid_folder'), os.environ.get('patch_size'), os.environ.get('arch'), os.environ.get('encoder'), os.environ.get('name_weights')
	patch_size = utils.os_environ(patch_size, 512, 'int')
	gpu = 	utils.os_environ(gpu, 0, 'int')
	gpu = [gpu]
	batch_size = utils.os_environ(batch_size, 30, 'int')
	valid_metrics = main(level, patch_size, name_dataset, valid_folder, batch_size, arch, encoder, name_weights, gpu)

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			     	
        	
        
        
        
