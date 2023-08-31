from torch.utils.data import Dataset, DataLoader
import os 
import random
import torch
import linecache 
#import matplotlib.pyplot as plt
import numpy as np
#from skimage import io
import sys 
#from PIL import Image
import torchvision.transforms as T
from skimage import io 
import data_utils as utils
from visu_slides import Files, Slides, Tumors
sys.path.append('/mnt/home.stud/laposben/Documents/Augmentation-PyTorch-Transforms-master')
#import myTransforms


#from pympler import asizeof

#############################################################################################################################################################################################################################
### new class built to train the networks used to predict the test slides. It uses many different negative patches in the course of the training, by selecting randomly, according to a fixed seed, the negative patches. ###
#############################################################################################################################################################################################################################

#only for the test training 
class Camelyon16Dataset_openslide_negrand(Dataset):

	def __init__(self, level, patch_size, transform=None, transform_nomask=None):
	
		self.level = level
		self.patch_size = patch_size 
		self.transform = transform 
		self.transform_nomask = transform_nomask
		
		self.path = f'/datagrid/personal/laposben/f_segm/dataset/{level}/{patch_size}'
		self.negs = self.__get_negs__()
		self.num_pos = self.__len__() // 2
		self.epoch = 0 #increase this number during the training to change the list of negative patches, using a callback and the method on_train_epoch_end. It should work 
		#self.curr_idx = 0 #to know when to switch epoch 
		self.f = Files()
		
		
	def __len__(self):
		
		num_pos = 0
		with open(os.path.join(self.path, 'images_pos.txt'), 'r') as R:
			num_pos += len(R.readlines())
		return 2 * num_pos
		
	def __getitem__(self, idx):
	
		assert isinstance(idx, int)
		self.curr_idx += 1
		if idx < self.num_pos:
			filename = os.path.join(self.path, 'images_pos.txt')
			img_line = linecache.getline(filename, idx + 1)
		else: 
			filename = os.path.join(self.path, 'images_neg.txt')
			img_line = linecache.getline(filename, self.negs[self.epoch][idx - self.num_pos])
		#if self.curr_idx == 2 * self.num_pos: self.epoch, self.curr_idx = self.epoch + 1, 0 #reset the curr_idx and switch epoch
		#print('self.curr_idx and self.epoch', self.curr_idx, self.epoch)
		img_id, img_label = img_line.split(";")[0], float(img_line.split(";")[1])
		id_tumor, row, col = img_id.split('_')[1], int(img_id.split('_')[2]), int(img_id.split('_')[3])
		h, w = int(row * self.patch_size), int(col * self.patch_size)
		tumor_name = 'tumor_' + id_tumor 
		path_tumor, path_mask = self.f.get_path_tumor(int(id_tumor)), self.f.get_path_mask(int(id_tumor))
		tum = Tumors(path_tumor, path_mask)		
		new_h, new_w, height, width = utils.add_perturbation((h,w), self.patch_size, [tum.get_dim(self.level, self.patch_size, 'THUMB')[1], tum.get_dim(self.level, self.patch_size, 'THUMB')[0]])
		
		if img_label > 0:
			image, mask = tum.get_region((new_h, new_w), self.level, size = (height, width))
			mask = mask[:,:,0]
			mask[mask == 1] = 255 #line added after, maybee it will change something since. To ensure there will be one after the conversion with ToTensor()		
		else:
			image, mask = tum.get_region((new_h, new_w), self.level, size = (height, width))
			mask = np.zeros((self.patch_size, self.patch_size, 1)) 
			
		if image.shape[0] != self.patch_size or image.shape[1] != self.patch_size: #add padding if the tile has not the good dimension because it is close to a border
			padding_color = self.f.get_background_pad(int(id_tumor))
			original_height, original_width, _ = image.shape
			pad_height, pad_width = self.patch_size - original_height, self.patch_size - original_width
			image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
			for channel in range(3):
				image[:,:,channel][image[:,:,channel] == 0] = padding_color[channel]
			if img_label > 0:
				mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')

		sample = {'image': image, 'mask': mask}
		
		if self.transform:
			sample['image'], sample['mask'] = self.transform(sample['image']), self.transform(sample['mask'])
		if self.transform_nomask:
			sample['image'] = self.transform_nomask(sample['image'])
						
		return sample 
		
		
		
	def __get_negs__(self, random_seed = 42, max_epochs = 100):
		
		with open(os.path.join(self.path, 'images_neg.txt'), 'r') as R:
			num_neg = len(R.readlines())
		len_neg = self.__len__() // 2
		random.seed(random_seed)
		negs_idx = [k for k in range(1, num_neg +1)]
		negs = []
		for _ in range(max_epochs): #####CARE: here I put the max_epochs as 100 because during the training I never excedded this value, but in case this changes I must change this value also. Be careful 
			negs.append(random.sample(negs_idx, len_neg))	
		random.seed() #so the rest of the algorithm remains random, idk if it is a good practice to do so
		return negs	

#############################################################################################################################################################################################################################
### new class built to train the networks used to predict the folder slides. It uses many different negative patches in the course of the training, by selecting randomly, according to a fixed seed, the negative patches. ###
#############################################################################################################################################################################################################################


class Camelyon16Dataset_openslide_negrand_validset(Dataset): 

	def __init__(self, level, patch_size, name_dataset, folder_valid, valid, test = False, transform=None, transform_nomask=None): 
	
		self.path = f'/datagrid/personal/laposben/f_segm/dataset/{level}/{patch_size}/custom_dataset/{name_dataset}'
		
		
		self.folder_valid = folder_valid 
		self.valid = valid #Boolean 
		self.transform = transform 
		self.transform_nomask = transform_nomask
		
		self.level = level
		self.patch_size = patch_size
		self.len = self.__get_len_folders__() #len de tous les folders sauf le valid ou alors que le valid si valid=True
		self.len_pos, self.len_pos_dic = self.__get_len_pos_folders__()#meme chose aue pour self.len mais on a queles nums pos 
		self.len_neg = [self.len[k] - self.len_pos[k] for k in range(len(self.len))] #nombre de negs a puiser par folder
		self.f = Files()
		
		self.negs = self.__get_negs__()
		self.epoch = 0 #increase this number during the training to change the list of negative patches, using a callback and the method on_train_epoch_end. It should work 


	def __len__(self):
	
		num = 0
		for n in self.len:
			num += n
		return num

	def __getitem__(self, idx):
	
		if torch.is_tensor(idx):
		   	idx = idx.tolist()
		
		if isinstance(idx, int):
			folder, idx_folder = self.__get_folder__(idx) #cette fonction me dit dans quel folder je dois piocher plus l'indice de la ligne, donc c'est parfait !
			if idx_folder - 1 < self.len_pos_dic[folder]:
				filename = os.path.join(self.path, f'pos_{folder}.txt')
				img_line = linecache.getline(filename, idx_folder)
			else:
				filename = os.path.join(self.path, f'neg_{folder}.txt')
				img_line = linecache.getline(filename, self.negs[self.epoch][folder][idx_folder - self.len_pos_dic[folder] - 1])

			img_id, img_label = img_line.split(";")[0], float(img_line.split(";")[1])
			id_tumor, row, col = img_id.split('_')[1], int(img_id.split('_')[2]), int(img_id.split('_')[3])
			h, w = int(row * self.patch_size), int(col * self.patch_size)
			tumor_name = 'tumor_' + id_tumor 
			path_tumor, path_mask = self.f.get_path_tumor(int(id_tumor)), self.f.get_path_mask(int(id_tumor))
			tum = Tumors(path_tumor, path_mask)
				#self.tif_files[tumor_name + '_size'] = [tum.get_dim(self.level, self.patch_size, 'THUMB')[1], tum.get_dim(self.level, self.patch_size, 'THUMB')[0]] #height, width 
			
			#new_h, new_w, height, width = utils.add_perturbation((h,w), self.patch_size, [tum.get_dim(self.level, self.patch_size, 'THUMB')[1], tum.get_dim(self.level, self.patch_size, 'THUMB')[0]]) #################
			new_h, new_w, height, width = h, w, 512, 512
			if img_label > 0:
				image, mask = tum.get_region((new_h, new_w), self.level, size = (height, width))
				mask = mask[:,:,0]
				mask[mask == 1] = 255 #line added after, maybee it will change something since. To ensure there will be one after the conversion with ToTensor()
				
			else:
				image, mask = tum.get_region((new_h, new_w), self.level, size = (height, width))
				mask = np.zeros((self.patch_size, self.patch_size, 1)) 
				
			if image.shape[0] != self.patch_size or image.shape[1] != self.patch_size: #add padding if the tile has not the good dimension because it is close to a border
				padding_color = self.f.get_background_pad(int(id_tumor))
				original_height, original_width, _ = image.shape
				pad_height, pad_width = self.patch_size - original_height, self.patch_size - original_width
				image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
				for channel in range(3):
					image[:,:,channel][image[:,:,channel] == 0] = padding_color[channel]
				if img_label > 0:
					mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
	
			sample = {'image': image, 'mask': mask}
			
			if self.transform:
				sample['image'], sample['mask'] = self.transform(sample['image']), self.transform(sample['mask'])
			if self.transform_nomask:
				sample['image'] = self.transform_nomask(sample['image'])
			
		return sample 

	def __get_len_folders__(self):
	
		lens = []
		if self.valid:
			folders = [self.folder_valid]
		else: 
			folders = [k for k in range(1,6) if k != self.folder_valid]
			
		for folder in folders:
			filename = os.path.join(self.path, f'{folder}.txt')
			with open(filename, 'r') as F:
				lens.append(len(F.readlines()))
		return lens	
		
	def __get_len_pos_folders__(self):
	
		lens = []
		lens_dic = {}
		if self.valid:
			folders = [self.folder_valid]
		else: 
			folders = [k for k in range(1,6) if k != self.folder_valid]
			
		for folder in folders:
			filename = os.path.join(self.path, f'pos_{folder}.txt')
			with open(filename, 'r') as F:
				lnt = len(F.readlines())
				lens.append(lnt)
				lens_dic[folder] = lnt
		return lens, lens_dic
	    
	def __get_folder__(self, idx):
		
		if self.valid:
			return self.folder_valid, idx + 1	
		idx_line = idx + 1
		cumulative_sum = np.cumsum(self.len)
		for i in range(len(cumulative_sum)): 
			if idx_line <= cumulative_sum[i]:
				pos = i
				folder = i + 1
				break 
		if pos > 0: 
			idx_in_folder = idx_line - cumulative_sum[pos - 1]
		else: 
			idx_in_folder = idx_line 	
		if self.folder_valid <= folder:
				folder += 1	
		return folder, idx_in_folder

	def __get_negs__(self, random_seed = 42, max_epochs = 100):
		
		negs_idx = []
		for k in range(len(self.len_neg)):
			negs_idx.append([i for i in range(1, self.len_neg[k] + 1)]) #une liste contenant les indices que je peux prendre parmis les differents folder 
		random.seed(random_seed)
		negs = [] #les indices effectivement choisis au cours des epochs 
		for _ in range(max_epochs): #####CARE: here I put the max_epochs as 100 because during the training I never excedded this value, but in case this changes I must change this value also. Be careful 
			negs_ep = {}
			k = 0
			for folder in list(self.len_pos_dic.keys()):
				negs_ep[folder] = random.sample(negs_idx[k], self.len_neg[k])
				k += 1
			negs.append(negs_ep)
		random.seed() #so the rest of the algorithm remains random, idk if it is a good practice to do so
		return negs



#################################################################################################
### first class that used openslide to create the snamples instead of loading presaved images ###
#################################################################################################


class Camelyon16Dataset_openslide(Dataset): 

	def __init__(self, level, tile_size, name_dataset, folder_valid, valid, test=False, transform=None, transform_nomask=None, mytransforms=None, preprocessed = False): 
	
		#self.path_to_dir = os.path.join(f'/local/temporary/f_segm/dataset/{level}/{tile_size}/custom_dataset', name_dataset) #to custom_dataset folder  ################################
		self.path_to_dir = os.path.join(f'/datagrid/personal/laposben/f_segm/dataset/{level}/{tile_size}/custom_dataset', name_dataset)
		
		self.transform = transform
		self.mytransforms = mytransforms
		self.transform_nomask = transform_nomask
		
		self.folder_valid = folder_valid 
		self.valid = valid #Boolean 
		self.test = test
		self.preprocessed = preprocessed
		
		self.level = level
		self.tile_size = tile_size
		self.len = self.__get_len_folders__()
		self.f = Files()
		self.tif_files = {}

	def __len__(self):
	
		num = 0
		for n in self.len:
			num += n
		return num

	def __getitem__(self, idx):
	
		if torch.is_tensor(idx):
		   	idx = idx.tolist()
		
		if isinstance(idx, int):
			folder, idx_folder = self.__get_folder__(idx)
			filename = os.path.join(self.path_to_dir, f'{folder}.txt')
			img_line = linecache.getline(filename, idx_folder)
			img_id, img_label = img_line.split(";")[0], float(img_line.split(";")[1])
			id_tumor, row, col = img_id.split('_')[1], int(img_id.split('_')[2]), int(img_id.split('_')[3])
			tumor_name = 'tumor_' + id_tumor 
			assert not(self.preprocessed), 'stain-normalization not yet implemented'
			path_tumor, path_mask = self.f.get_path_tumor(int(id_tumor)), self.f.get_path_mask(int(id_tumor))
			tum = Tumors(path_tumor, path_mask)

			if img_label > 0:
				image, mask = tum.get_patch(self.level, self.tile_size, row, col)
				mask = mask[:,:,0]
				mask[mask == 1] = 255 #line added after, maybee it will change something since 
				
			else:
				image = tum.get_patch(self.level, self.tile_size, row, col)[0]
				mask = np.zeros((self.tile_size, self.tile_size, 1)) 
				
			if image.shape[0] != self.tile_size or image.shape[1] != self.tile_size: #add padding if the tile has not the good dimension because it is close to a border
				padding_color = self.f.get_background_pad(int(id_tumor))
				original_height, original_width, _ = image.shape
				pad_height, pad_width = self.tile_size - original_height, self.tile_size - original_width
				image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
				for channel in range(3):
					image[:,:,channel][image[:,:,channel] == 0] = padding_color[channel]
				if img_label > 0:
					mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
	
			sample = {'image': image, 'mask': mask}
			
			if self.transform:
				sample['image'], sample['mask'] = self.transform(sample['image']), self.transform(sample['mask'])
			if self.transform_nomask:
				sample['image'] = self.transform_nomask(sample['image'])
			if self.mytransforms:
				for function, proba in self.mytransforms: 
					p = np.random.rand(0,1)
					if p <= proba and not(str(function).split('(')[0]) == 'HEDJitter': 
						sample['image'], sample['mask'] = function(sample['image']), function(sample['mask'])
					elif p <= proba and str(function).split('(')[0] == 'HEDJitter':
						sample['image'] = function(sample['image'])
			
		return sample 
	
############# I think the best thing to do is to create another class for noow and if it works well, I can change
###nextime### this class to the new. The idea is to use the .get_region instead of patches that let us have more
############# liberty - I need to test before if the get.regions is as fast as the patch, it should be just the same. Then, I can just had a perturbation to the coordonates of the patches, to extract different patches at each time. In cancerous regions, it will just act as a translation of existing patches so I will increase the variety of the patches, and for neutral patches it offers to the network images he has not seen before so it gives more data variety to decrease overfitting and to try improving the performance, as I saw that the resne101 can increases the performance, but mostly on the train data set so I need to change to reduce overfitting. Just adding this perturbation should be enough. I can also study elastic transformation even If it do not know how well it can do, I think I should study this and implement it using parameters of low value. RandomApply and RandomChoice seem also to be interesting. So tomorrow I do this, I finish the Evaluation Froc file, finally and test it using different values for erode and new_contours. Then I can finally start to produce final results for the level 0 and go to the level 1. I also have to fasten the test pipeline to producethe inference. 


	def __get_len_folders__(self):
	
		lens = []
		if self.valid:
			folders = [self.folder_valid]
		elif self.test:
			folders = [k for k in range(1,6)]
		else: 
			folders = [k for k in range(1,6) if k != self.folder_valid]
			
		for folder in folders:
			filename = os.path.join(self.path_to_dir, f'{folder}.txt')
			with open(filename, 'r') as F:
				lens.append(len(F.readlines()))
		return lens
	    
	def __get_folder__(self, idx):
		
		if self.valid:
			return self.folder_valid, idx + 1
			
		idx_line = idx + 1
		cumulative_sum = np.cumsum(self.len)
		for i in range(len(cumulative_sum)): 
			if idx_line <= cumulative_sum[i]:
				pos = i
				folder = i + 1
				break 
		if pos > 0: 
			idx_in_folder = idx_line - cumulative_sum[pos - 1]
		else: 
			idx_in_folder = idx_line 
		
		if not(self.test):
			if self.folder_valid <= folder:
				folder += 1
		
		return folder, idx_in_folder


###########################################
### class with random crop perturbation ###
###########################################

class Camelyon16Dataset_openslide_rand(Dataset): 

	def __init__(self, level, tile_size, name_dataset, folder_valid, valid, test = False, transform=None, transform_nomask=None, mytransforms=None, preprocessed = False): 
	
		#self.path_to_dir = os.path.join(f'/local/temporary/f_segm/dataset/{level}/{tile_size}/custom_dataset', name_dataset) #to custom_dataset folder  ################################
		self.path_to_dir = os.path.join(f'/datagrid/personal/laposben/f_segm/dataset/{level}/{tile_size}/custom_dataset', name_dataset)
		
		self.transform = transform
		self.mytransforms = mytransforms
		self.transform_nomask = transform_nomask
		
		self.folder_valid = folder_valid 
		self.valid = valid #Boolean 
		self.test = test 
		self.preprocessed = preprocessed
		
		self.level = level
		self.tile_size = tile_size
		self.len = self.__get_len_folders__()
		self.f = Files()
		self.tif_files = {}

	def __len__(self):
	
		num = 0
		for n in self.len:
			num += n
		return num

	def __getitem__(self, idx):
	
		if torch.is_tensor(idx):
		   	idx = idx.tolist()
		
		if isinstance(idx, int):
			folder, idx_folder = self.__get_folder__(idx)
			filename = os.path.join(self.path_to_dir, f'{folder}.txt')
			img_line = linecache.getline(filename, idx_folder)
			img_id, img_label = img_line.split(";")[0], float(img_line.split(";")[1])
			id_tumor, row, col = img_id.split('_')[1], int(img_id.split('_')[2]), int(img_id.split('_')[3])
			h, w = int(row * self.tile_size), int(col * self.tile_size)
			tumor_name = 'tumor_' + id_tumor 
			assert not(self.preprocessed), 'stain-normalization not yet implemented'
			path_tumor, path_mask = self.f.get_path_tumor(int(id_tumor)), self.f.get_path_mask(int(id_tumor))
			tum = Tumors(path_tumor, path_mask)
				#self.tif_files[tumor_name + '_size'] = [tum.get_dim(self.level, self.tile_size, 'THUMB')[1], tum.get_dim(self.level, self.tile_size, 'THUMB')[0]] #height, width 
			
			new_h, new_w, height, width = utils.add_perturbation((h,w), self.tile_size, [tum.get_dim(self.level, self.tile_size, 'THUMB')[1], tum.get_dim(self.level, self.tile_size, 'THUMB')[0]])
			print(new_h, new_w)
			if img_label > 0:
				image, mask = tum.get_region((new_h, new_w), self.level, size = (height, width))
				mask = mask[:,:,0]
				mask[mask == 1] = 255 #line added after, maybee it will change something since. To ensure there will be one after the conversion with ToTensor()
				
			else:
				image, mask = tum.get_region((new_h, new_w), self.level, size = (height, width))
				mask = np.zeros((self.tile_size, self.tile_size, 1)) 
				
			if image.shape[0] != self.tile_size or image.shape[1] != self.tile_size: #add padding if the tile has not the good dimension because it is close to a border
				padding_color = self.f.get_background_pad(int(id_tumor))
				original_height, original_width, _ = image.shape
				pad_height, pad_width = self.tile_size - original_height, self.tile_size - original_width
				image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
				for channel in range(3):
					image[:,:,channel][image[:,:,channel] == 0] = padding_color[channel]
				if img_label > 0:
					mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
	
			sample = {'image': image, 'mask': mask}
			
			if self.transform:
				sample['image'], sample['mask'] = self.transform(sample['image']), self.transform(sample['mask'])
			if self.transform_nomask:
				sample['image'] = self.transform_nomask(sample['image'])
			if self.mytransforms:
				for function, proba in self.mytransforms: 
					p = np.random.rand(0,1)
					if p <= proba and not(str(function).split('(')[0]) == 'HEDJitter': 
						sample['image'], sample['mask'] = function(sample['image']), function(sample['mask'])
					elif p <= proba and str(function).split('(')[0] == 'HEDJitter':
						sample['image'] = function(sample['image'])
			
		return sample 
	
############# I think the best thing to do is to create another class for noow and if it works well, I can change
###nextime### this class to the new. The idea is to use the .get_region instead of patches that let us have more
############# liberty - I need to test before if the get.regions is as fast as the patch, it should be just the same. Then, I can just had a perturbation to the coordonates of the patches, to extract different patches at each time. In cancerous regions, it will just act as a translation of existing patches so I will increase the variety of the patches, and for neutral patches it offers to the network images he has not seen before so it gives more data variety to decrease overfitting and to try improving the performance, as I saw that the resne101 can increases the performance, but mostly on the train data set so I need to change to reduce overfitting. Just adding this perturbation should be enough. I can also study elastic transformation even If it do not know how well it can do, I think I should study this and implement it using parameters of low value. RandomApply and RandomChoice seem also to be interesting. So tomorrow I do this, I finish the Evaluation Froc file, finally and test it using different values for erode and new_contours. Then I can finally start to produce final results for the level 0 and go to the level 1. I also have to fasten the test pipeline to producethe inference. 


	def __get_len_folders__(self):
	
		lens = []
		if self.valid:
			folders = [self.folder_valid]
		if self.test:
			folders = [k for k in range(1,6)]
		else: 
			folders = [k for k in range(1,6) if k != self.folder_valid]
			
		for folder in folders:
			filename = os.path.join(self.path_to_dir, f'{folder}.txt')
			with open(filename, 'r') as F:
				lens.append(len(F.readlines()))
		return lens
	    
	def __get_folder__(self, idx):
		
		if self.valid:
			return self.folder_valid, idx + 1
			
		idx_line = idx + 1
		cumulative_sum = np.cumsum(self.len)
		for i in range(len(cumulative_sum)): 
			if idx_line <= cumulative_sum[i]:
				pos = i
				folder = i + 1
				break 
		if pos > 0: 
			idx_in_folder = idx_line - cumulative_sum[pos - 1]
		else: 
			idx_in_folder = idx_line 
		
		if not self.test:
			if self.folder_valid <= folder:
				folder += 1
		
		return folder, idx_in_folder


class Camelyon16Dataset_v2(Dataset): 

	def __init__(self, level, name_dataset, folder_valid, valid, transform=None, transform_nomask=None, mytransforms=None, preprocessed = False): 
	
		self.path_to_dir = os.path.join(f'/local/temporary/patches_lv{level}/custom_dataset', name_dataset) #to custom_dataset folder 
		
		self.transform = transform
		self.mytransforms = mytransforms
		self.transform_nomask = transform_nomask
		
		self.folder_valid = self.folder_valid 
		self.valid = valid
		self.preprocessed = preprocessed
		
		self.len = self.__get_len_folders__()

	def __len__(self):
	
		num = 0
		for n in self.len:
			num += n
		return num

	def __getitem__(self, idx):
	
		if torch.is_tensor(idx):
		   	idx = idx.tolist()
		
		if isinstance(idx, int):
			folder, idx_folder = self.__get_folder__(idx)
			filename = os.path.join(self.path_to_dir, f'{folder}.txt')
			img_line = linecache.getline(filename, idx_folder)
			img_id, img_label = img_line.split(";")[0], img_line.split(";")[1]
			if self.preprocessed:
				img_path = os.path.join(self.root,'macenko_patch',img_id + '.png')
			else:
				img_path = os.path.join(self.root,'patch',img_id + '.png')
				
			image = np.transpose(io.imread(img_path)[:,:,:3], (2,0,1)) #so the channels dim appears first, we dont get the alpha channel
			image = torch.tensor(image)
			if float(img_label) > 0: #presence of tumors
				mask_path = os.path.join(self.root,'label',img_id + '.png')
				#mask = np.transpose(io.imread(mask_path)[:,:,0] / 255, (2,0,1)) #remove alpha channel, only keep one channel and obtain 0 1
				mask = np.expand_dims(io.imread(mask_path)[:,:,0] / 255, axis = 0)
				mask = torch.tensor(mask)
			else : 
				mask = torch.zeros((1,256,256))
			sample = {'image': image, 'mask': mask}
			
			if self.transform:
				sample['image'], sample['mask'] = self.transform(sample['image']), self.transform(sample['mask'])
			if self.transform_nomask:
				sample['image'] = self.transform_nomask(sample['image'])
			if self.mytransforms:
				for function, proba in self.mytransforms: 
					p = np.random.rand(0,1)
					if p <= proba and not(str(function).split('(')[0]) == 'HEDJitter': 
						sample['image'], sample['mask'] = function(sample['image']), function(sample['mask'])
					elif p <= proba and str(function).split('(')[0] == 'HEDJitter':
						sample['image'] = function(sample['image'])
			
		return sample 
	
	def __get_len_folders__(self):
	
		lens = []
		if self.valid:
			folders = [self.folder_valid]
		else: 
			folders = [k for k in range(1,6) and k != self.folder_valid]
			
		for folder in folders:
			filename = os.path.join(self.path_to_dir, f'{folder}.txt')
			with open(filename, 'r') as F:
				lens.append(len(F.readlines()))
		return lens
	    
	def __get_folder__(self, idx):
	
		cumulative_sum = np.cumsum(self.len)
		for i in range(len(cumulative_sum)): 
			if idx < cumulative_sum[i]:
				pos = i
				break 
		
		folder = pos + 1 
		folder = folder + self.valid_folder <= folder
		if pos > 0:
			idx_real = idx + 1 - cumulative_sum[pos - 1]
		else: 
			idx_real = idx + 1

		return folder, idx_real + 1 





class Camelyon16Dataset(Dataset): 

	def __init__(self, root, root_dir, name_txt, transform=None, transform_nomask=None, mytransforms=None, preprocessed = False): 
		self.root = root
		self.root_dir = root_dir 
		
		self.transform = transform
		self.mytransforms = mytransforms
		self.transform_nomask = transform_nomask
		
		self.name_txt = name_txt
		self.preprocessed = preprocessed

	def __len__(self):
	    	filename = os.path.join(self.root_dir,self.name_txt)
	    	with open(filename, 'r') as file:
	    		num_lines = len(file.readlines())
	    	return num_lines

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
		   	idx = idx.tolist()
		
		if isinstance(idx, int):
			filename = os.path.join(self.root_dir,self.name_txt)
			img_line = linecache.getline(filename, idx + 1)
			img_id, img_label = img_line.split(";")[0], img_line.split(";")[1]
			if self.preprocessed:
				img_path = os.path.join(self.root,'macenko_patch',img_id + '.png')
			else:
				img_path = os.path.join(self.root,'patch',img_id + '.png')
		
			image = load_image(img_path, self, filename)
			#image = np.transpose(io.imread(img_path)[:,:,:3], (2,0,1)) #so the channels dim appears first, we dont get the alpha channel				
			image = torch.tensor(image)
			if float(img_label) > 0: #presence of tumors
				mask_path = os.path.join(self.root,'label',img_id + '.png')
				#mask = np.transpose(io.imread(mask_path)[:,:,0] / 255, (2,0,1)) #remove alpha channel, only keep one channel and obtain 0 1
				mask = np.expand_dims(io.imread(mask_path)[:,:,0] / 255, axis = 0)
				mask = torch.tensor(mask)
			else : 
				mask = torch.zeros((1,256,256))
			sample = {'image': image, 'mask': mask}
			
			if self.transform:
				sample['image'], sample['mask'] = self.transform(sample['image']), self.transform(sample['mask'])
			if self.transform_nomask:
				sample['image'] = self.transform_nomask(sample['image'])
			if self.mytransforms:
				for function, proba in self.mytransforms: 
					p = np.random.rand(0,1)
					if p <= proba and not(str(function).split('(')[0]) == 'HEDJitter': 
						sample['image'], sample['mask'] = function(sample['image']), function(sample['mask'])
					elif p <= proba and str(function).split('(')[0] == 'HEDJitter':
						sample['image'] = function(sample['image'])
			
		return sample 
		
##########################		
### to test the RAM bug###
##########################

class Camelyon16Dataset_test(Dataset): 

	def __init__(self, root, root_dir, name_txt, transform=None, transform_nomask=None, mytransforms=None, preprocessed = False): 
		self.root = root
		self.root_dir = root_dir 
		
		self.transform = transform
		self.mytransforms = mytransforms
		self.transform_nomask = transform_nomask
		
		self.name_txt = name_txt
		self.preprocessed = preprocessed

	def __len__(self):
		files = []
		files += [each for each in os.listdir(self.root) if each.endswith('.png')]
		return len(files)
	 
	#@profile
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
		   	idx = idx.tolist()
		
		if isinstance(idx, int):
			files = []
			files += [each for each in os.listdir(self.root) if each.endswith('.png')]
	
			img = files[idx]
			img_path = os.path.join(self.root, img)
			image = io.imread(img_path)[:,:,:3]
			#image = np.transpose(io.imread(img_path)[:,:,:3], (2,0,1))
			#image = torch.tensor(image)
			mask = np.zeros((1,256,256))
			sample = {'image': image, 'mask': mask}

			if self.transform:
				sample['image'], sample['mask'] = self.transform(sample['image']), self.transform(sample['mask'])
			if self.transform_nomask:
				sample['image'] = self.transform_nomask(sample['image'])
			if self.mytransforms:
				for function, proba in self.mytransforms: 
					p = np.random.rand(0,1)
					if p <= proba and not(str(function).split('(')[0]) == 'HEDJitter': 
						sample['image'], sample['mask'] = function(sample['image']), function(sample['mask'])
					elif p <= proba and str(function).split('(')[0] == 'HEDJitter':
						sample['image'] = function(sample['image'])
		return sample 



def load_image(img_path, obj, filename): 
	try:
		image = np.transpose(io.imread(img_path)[:,:,:3], (2,0,1))
		return image
	except:
		idx = np.random.randint(0, obj.__len__())
		img_line = linecache.getline(filename, idx + 1)
		img_id, img_label = img_line.split(";")[0], img_line.split(";")[1]
		img_path = os.path.join(obj.root,'macenko_patch',img_id + '.png')	
		return load_image(img_path, obj, filename)

class Camelyon16Dataset_v0(Dataset): 

	def __init__(self, root_dir, name_txt, transform=None): 
	
		self.root_dir = root_dir 
		self.transform = transform
		self.name_txt = name_txt

	def __len__(self):
	    	filename = os.path.join(self.root_dir,self.name_txt)
	    	with open(filename, 'r') as file:
	    		num_lines = len(file.readlines())
	    	return num_lines

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
		   	idx = idx.tolist()
		
		if isinstance(idx, int):
			filename = os.path.join(self.root_dir,self.name_txt)
			img_line = linecache.getline(filename, idx + 1)
			img_id, img_label = img_line.split(";")[0], img_line.split(";")[1]
			img_path = os.path.join(self.root_dir,'patch',img_id + '.png')
			image = np.transpose(io.imread(img_path)[:,:,:3], (2,0,1)) #so the channels dim appears first, we dont get the alpha channel
			if float(img_label) > 0: #presence of tumors
				mask_path = os.path.join(self.root_dir,'label',img_id + '.png')
				#mask = np.transpose(io.imread(mask_path)[:,:,0] / 255, (2,0,1)) #remove alpha channel, only keep one channel and obtain 0 1
				mask = np.expand_dims(io.imread(mask_path)[:,:,0] / 255, axis = 0)
			else : 
				mask = np.zeros((1,256,256))
			sample = {'image': image, 'mask': mask}
			
		if isinstance(idx, list): 
			sample = []
			for idxs in idx : 
				filename = os.path.join(self.root_dir,self.name_txt)
				img_lime = linecache.getline(filename, idxs + 1)
				img_id, img_label = img_line.split(";")[0], img_line.split(";")[1]
				img_path = os.path.join(self.root_dir,'patch',img_id + '.png')
				image = np.transpose(io.imread(img_path)[:,:,:3], (2,0,1)) 
				if float(img_label) > 0: #presence of tumors
					mask_path = os.path.join(self.root_dir,'label',img_id + '.png')
					#mask = np.transpose(io.imread(mask_path)[:,:,0] / 255, (2,0,1))
					mask = np.expand_dims(io.imread(mask_path)[:,:,0] / 255, axis = 0)
				else : 
					mask = np.zeros((256,256,1))
				sample.append({'image': image, 'mask': mask})
				
		return sample 
'''
import psutil
	

def main():
	root, root_dir = '/local/temporary/macenko_patch', '/datagrid/personal/laposben/f_segm/dataset/2/256/custom_dataset/cut_[-1.0, 1.0]_numpos_25001_factor_0.5'
	batch_size = 16
	n_cpu = 128
	transform, transform_nomask = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.2), T.RandomVerticalFlip(0.2)]), T.Compose([T.ColorJitter(brightness = 0.2, contrast = 0.3, saturation = 0.3, hue = 0.04), T.GaussianBlur(kernel_size = 3, sigma= 0.8)])
	train_dataset = Camelyon16Dataset_openslide(level = 0, tile_size = 512, name_dataset = 'cut_[-1.0, 1.0]_numpos_86416_factor_1.0', folder_valid = 1, valid = False, transform = transform, transform_nomask = transform_nomask) # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) https://towardsdatascience.com/data-augmentations-in-torchvision-5d56d70c372e  ########################################Put the rand
	#train_dataset = Camelyon16Dataset_test(root, root_dir, '1.txt', transform = transform, transform_nomask =  transform_nomask, mytransforms = None, preprocessed = False)
	#valid_dataset = Camelyon16Dataset_v0(level = 0, tile_size = 512, name_dataset = 'cut_[-1.0, 1.0]_numpos_86416_factor_1.0', folder_valid = 1, valid = True, transform = T.ToTensor()) #########no rand here
	train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=0, persistent_workers = False) ########works 0 and not n_cpu
	#valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size,shuffle = False, num_workers=n_cpu // 2, persistent_workers = False) #####used to have 4 as batch_size, think I did it because of GPU memory constraints 
	i = 0
	for batch in iter(train_dataloader):
		print(batch['image'][0,0,0,0])
		i += 1
		ram_usage = psutil.virtual_memory().used / 1024**3
		print(f"RAM Usage: {ram_usage} GB")
		print(sys.getsizeof(train_dataset), 'memory of train_dataset')
		#gc.collect()
		if i == 100:
			break
				
if __name__ == "__main__":
	
	main()
'''	
	
	
	
	
	
	
	
	
	
	





