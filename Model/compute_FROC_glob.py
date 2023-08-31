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
from test import build_fast_predictions, Save_Test
import Evaluation_FROC_original 

sys.path.append('/mnt/home.stud/laposben/Documents/Augmentation-PyTorch-Transforms-master')
import myTransforms
from NN import SegModel, MaxLossBatchCallback, SaveBestModel, LoadBestModel

def main(name_exp_preds, level, valid_folder, name_dataset, name_exp, patch_size, level_pred, Test = False): #level_pred: level ou la prediction a ete faite, level: level ou la prediction a ete construite 

	for th in [0.5, 0.7, 0.9]: #need to lower to 0.7 maybe because 0.9 is too high for UNETcombineloss
		for erode_iter in [1,2]:
			print('th:' , th, 'erode_iter: ', erode_iter)
			if not(Test):
				pred = build_fast_predictions()
				pred.compute_csv_all(name_exp_preds, th, level, name_dataset, name_exp, valid_folder, patch_size, level_pred, erode_iter)
			else:
				pred = Save_Test()
				pred.compute_csv_all(name_exp_preds, th, level, name_exp, patch_size, level_pred, erode_iter)
			resized = pow(2, level - level_pred)
			name_exp_FROC = name_exp + f';er:{erode_iter}'
			Evaluation_FROC_original.main(name_exp_FROC, valid_folder, level_pred, patch_size, name_exp_FROC, th, resized, Test = Test)
			
			
if __name__ == '__main__':
	
	#patch_size=512 level=3 name_dataset='cut_[-1.0, 1.0]_numpos_86416_factor_1.0' level_pred=0 name_exp_preds=UNETcombineloss valid_folder=5 name_exp=UNETcombinelossTESTtoerase python compute_FROC_glob.py
	name_exp_preds, level, valid_folder, name_dataset, name_exp, patch_size, level_pred = os.environ.get('name_exp_preds'), int(os.environ.get('level')), os.environ.get('valid_folder'), os.environ.get('name_dataset'), os.environ.get('name_exp'), os.environ.get('patch_size'), int(os.environ.get('level_pred'))
	#Test = utils.os_environ(Test, False, 'boolean')
	main(name_exp_preds, level, valid_folder, name_dataset, name_exp, patch_size, level_pred, Test = True)
	

	

	

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			     	
        	
        
        
        
