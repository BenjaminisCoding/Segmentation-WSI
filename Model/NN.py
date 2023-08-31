import os
import numpy as np
from tqdm import tqdm
#import matplotlib.pyplot as plt
#from skimage import io

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
#import gc ###################################################################
#%matplotlib inline

from pprint import pprint
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback

#%load_ext autoreload
#%autoreload 2

import sys
sys.path.append('/mnt/home.stud/laposben/Documents/Scripts/data')
from data_io import Camelyon16Dataset_openslide_rand, Camelyon16Dataset_openslide
#from data_io import Camelyon16Dataset_v0

sys.path.append('/mnt/home.stud/laposben/Documents/Augmentation-PyTorch-Transforms-master')
#import myTransforms

###############
### Network ###
###############


class SegModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, #**kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)   ####################Changed the loss function to incorporate cross-entropy to tackle over confidence. To calibrate the model. 
        self.loss_ce = smp.losses.SoftBCEWithLogitsLoss(reduction = 'mean', ignore_index = None, smooth_factor = 1e-2) 
        
        #custom 
        #self.current_batch = 0 #used to use it for the MaxLossBatchCallback
        self.best_dataset_iou = 0
        self.update_lr = 0 #attribute used to monitor how many times does the learning rate has decreased. 0 by default. used in the LoadBestModel callback
        #self.best_dataset_iou = kwargs['best_dataset_iou']
	
        self.valid_folder = kwargs['valid_folder']
        self.name_model = kwargs['name_model']
        self.save_path = kwargs['save_path']
        self.train_dataset = kwargs['train_dataset'] ###################### useful only when using negrand dataset class
        

    def forward(self, image):
        # normalize image here, maybee change the mean and std since we retrain the weights of the encoder, so need to compute the mean and std of my data 
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask


    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        #loss = self.loss_fn(logits_mask, mask)
        loss = 0.5 * self.loss_fn(logits_mask, mask) + 0.5 * self.loss_ce(logits_mask, mask) ############################################### here it is the crossentropy loss to use and not the NLL lose because the FPN does not have an activation layer at the end
        self.log(f'loss_{stage}', loss, on_step = True) #sync_dist=True)
        #self.log_dict(loss_dic, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid() ############################.detach()
        pred_mask = (prob_mask > 0.5).float()
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long().detach(), mode="binary")
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        per_image_recall = smp.metrics.recall(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_recall = smp.metrics.recall(tp, fp, fn, tn, reduction='micro')
        
        per_image_precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro') 

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_per_image_recall": per_image_recall,
            f"{stage}_dataset_recall": dataset_recall,
            f"{stage}_per_image_precision": per_image_precision,
            f"{stage}_dataset_precision": dataset_precision
        }
        
        self.log_dict(metrics, prog_bar=True)#, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
    	optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) ######################## change to 1e-4
    	#optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
    	#scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, verbose = True) ###############################step_size=10
    	#scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, verbose = True) ###################### changed to 8, was 10 because I am strarting with pretrained weights with 25 epochs
    	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience=20, threshold = 1e-4, verbose = True) #Might be better than programmed decreased learning rate, we shall see
    	return [optimizer], [{"scheduler": scheduler, 'monitor' : 'loss_valid'}]
    	#return [optimizer], [scheduler], 
    	     
            
#################
### Callbacks ###
#################
    
class MaxLossBatchCallback(Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0): #added h, j, k because he says 6 arguments are given
        #print(trainer.callback_metrics['val_loss'])
        #print(batch_idx)
        pass 
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        #if pl_module.current_batch % 10 == 0:
           # print(pl_module.current_batch)
        pass
        
    def on_train_epoch_end(self, trainer, pl_module):
        print(f'Epoch {trainer.current_epoch} is over')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        print(f'Validation Epoch {trainer.current_epoch} is over')
        
    def on_fit_end(self, trainer, pl_module):
        print('The training step is over')
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # get the validation loss for the current batch
        #print(outputs)
        pass     
        
        
class SaveBestModel(Callback):

	def on_validation_epoch_end(self, trainer, pl_module):
	
		if trainer.state.stage == "sanity_check": #Added so it no longer save the model when the sanity check is done
			return 
			
		metrics = trainer.callback_metrics
		print(metrics, 'metrics')
		valid_dataset_iou = metrics['valid_dataset_iou']
		if valid_dataset_iou > pl_module.best_dataset_iou:
			pl_module.best_dataset_iou = valid_dataset_iou
			state_dict = pl_module.state_dict()
			torch.save(state_dict, os.path.join(pl_module.save_path, pl_module.name_model + f'_val:{pl_module.valid_folder}.pt'))
			print(f'New model saved for the new performance {valid_dataset_iou}') 
		path_txt = os.path.join(pl_module.save_path, f'historic_metrics:{pl_module.name_model}:val{pl_module.valid_folder}')
		line = f'epochs:{trainer.current_epoch};metrics:{trainer.callback_metrics};update_lr:{pl_module.update_lr}\n' 
		if pl_module.current_epoch == 0 and not(os.path.exists(path_txt)): #if we continue the training, we do not want to erase the previous results, even though they are written in the logs file
			with open(path_txt, 'w') as W:
				pass
				#W.write(line)
		else:
			with open(path_txt, 'a') as A:
				A.write(line)
class SaveCheckPoint(Callback):

	def on_train_epoch_start(self, trainer, pl_module): #not end to be sure the trainer.current_epoch has been updated
	
		if trainer.current_epoch % 20 == 0 and trainer.current_epoch != 0:
			state_dict = pl_module.state_dict()
			torch.save(state_dict, os.path.join(pl_module.save_path, pl_module.name_model + f'val{pl_module.valid_folder};ep:{trainer.current_epoch}.pt')) ##################3to change when doing the test training because no val folder
	def on_fit_end(self, trainer, pl_module):
		
		state_dict = pl_module.state_dict()
		torch.save(state_dict, os.path.join(pl_module.save_path, pl_module.name_model + f'val{pl_module.valid_folder};ep:{trainer.current_epoch}.pt'))
	

class UpdateDataClass(Callback):

	def on_train_epoch_end(self, trainer, pl_module): 
		pl_module.train_dataset.epoch += 1
		print(pl_module.train_dataset.epoch)
			
class LoadBestModel(Callback):

	def __init__(self, start_lr):
		self.update_weights = 0 #how many times did this callback was called
		self.start_lr = start_lr
		
	def on_train_epoch_start(self, trainer, pl_module):
		
		current_lr = trainer.optimizers[0].param_groups[0]['lr']
		if self.start_lr > current_lr: #the lr has decreased
			pl_module.update_lr += 1
			pl_module.load_state_dict(torch.load(os.path.join(pl_module.save_path, pl_module.name_model + f'_val:{pl_module.valid_folder}.pt'))) #I hope it will indeed make the change to the pl_module, it should do it because of the "passing by reference"
			self.update_weights += 1
			self.start_lr = current_lr 
			print(f'Model loaded for the previous performance {pl_module.best_dataset_iou}')
'''		
if __name__ == "__main__":

	model = SegModel(arch = 'UNET', encoder_name='resnet34', in_channels=3, out_classes=1, save_path = '.', name_model = '.', valid_folder = 1)
	transform, transform_nomask = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.2), T.RandomVerticalFlip(0.2)]), T.Compose([T.ColorJitter(brightness = 0.2, contrast = 0.3, saturation = 0.3, hue = 0.04), T.GaussianBlur(kernel_size = 3, sigma= 0.8)])
	train_dataset = Camelyon16Dataset_openslide_rand(level = 0, tile_size = 512, name_dataset = 'cut_[-1.0, 1.0]_numpos_86416_factor_1.0', folder_valid = 1, valid = False, transform = transform, transform_nomask = transform_nomask)
	valid_dataset = Camelyon16Dataset_openslide(level = 0, tile_size = 512, name_dataset = 'cut_[-1.0, 1.0]_numpos_86416_factor_1.0', folder_valid = 1, valid = True, transform = T.ToTensor()) #########no rand here
	train_dataloader = DataLoader(train_dataset, batch_size = 25, shuffle = True, num_workers=128 // 2, persistent_workers = False) ########works 0 and not n_cpu
	valid_dataloader = DataLoader(valid_dataset, batch_size = 25,shuffle = False, num_workers=128 // 2, persistent_workers = False)
	i = 0
	for batch in tqdm(iter(train_dataloader)):
		model.shared_step(batch, 'train')
		i += 1
		if i == 10:
			break
		
'''			
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
