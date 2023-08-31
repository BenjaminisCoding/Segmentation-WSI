#from __future__ import division
#from PIL import Image

from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator

import os 
import cv2 as cv
#import linecache
#import random
import math
#from skimage import io 
import time 
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline

import data_utils as utils
	
class Files():

	def get_path_tumor(self, idx):
		idx_str = self.__idx_to_str__(idx)
		return '/datagrid/Medical/microscopy/CAMELYON16/training/tumor/' + idx_str + '.tif' ###################################
		
	def get_path_mask(self, idx):
		idx_str = self.__idx_to_str__(idx)
		if idx in self.__tumor__processed_before__(): return '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask/' + idx_str + '_Mask.tif'
		#else: return '/local/temporary/CAMELYON16/Mask/' + idx_str + '_Mask.tif'
		else: return '/datagrid/personal/laposben/Truth/Mask/' + idx_str + '_Mask.tif'
		
	def get_path_test(self, idx):
		idx_str = utils.idx_to_str(idx, Test = True)
		return '/datagrid/Medical/microscopy/CAMELYON16/testing/' + idx_str + '.tif'
		
	def get_path_normal(self, idx):
		idx_str = utils.idx_to_str(idx, Test = False, Norm = True)
		return '/datagrid/Medical/microscopy/CAMELYON16/training/normal/' + idx_str + '.tif'
		
	def get_path_tissue_mask(self, idx):
		idx_str = self.__idx_to_str__(idx)
		return '/datagrid/personal/laposben/tissue_mask/' + idx_str + '_tissue_mask.png' 
		#return '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask/' + idx_str + '_tissue_mask.png' 
		
	def get_path_thumb_masked(self, idx): 
		idx_str = self.__idx_to_str__(idx)
		return '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/' + idx_str + '_thumb.png' 
		
	def get_path_otsu_mask(self, idx):
		idx_str = self.__idx_to_str__(idx)
		with open('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tumortothreshold.txt', 'r') as R: 
			for line in R: 
				id_tumor, th = line.split(';')[0], line.split(';')[1].split('\n')[0]
				if id_tumor == idx_str: threshold = th
		return '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/' + idx_str + ' _ostu_threshold_' + threshold + '.png' 
	
	def get_path_target(self, idx): 
		idx_str = self.__idx_to_str__(idx)
		with open(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets', 'otsu_threshold_lv3.txt'), 'r') as F: 
			for line in F:
				line_id, beta = line.split(';')[0], float(line.split(';')[2].split('\n')[0])
				if line_id == idx_str:
					break		
		return '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets/target_' + idx_str + '.png', beta
		
	def __idx_to_str__(self, idx):
		if isinstance(idx, str): return idx 
		tens = len(str(idx))
		if tens == 1: return f'tumor_00{idx}'
		if tens == 2: return f'tumor_0{idx}'
		if tens == 3: return f'tumor_{idx}'

	def __tumor__processed_before__(self): 		
		return utils.get_original_tumors()
		
	def get_non_usable_tumors(self):
		return utils.get_non_usable_tumors()
	
	def get_len_dataset(self, d1 = True): #the dataset is quite tricky, because tumors from 0 to 70 amd tumor 111 belongs to d1, and the others to d2. So care when dealing with the dataset.
		return utils.len_datasets(d1)
		
	def get_idx_dataset(self, d1 = True):
	
		if d1:
			dataset = [k for k in range(1,71)]
			dataset.append(111)
			return dataset
		else:
			return [k for k in range(71, 111)]
	
	def get_background_pad(self, idx, Test = False): #to see how those values were computed, see the SangBag ipython file in the Model folder
		# idx here corresponds to the real label of the tumor, tumor_001 is labeled 1 as the first slide
		if Test:
			if idx in utils.dataset_test()[0]:
				return np.array([255, 255, 255])
			if idx in utils.dataset_test()[1]:
				return utils.get_pixels_avg(Test = True)[idx]
		else:
			if idx in self.get_idx_dataset(d1 = True):
				return np.array([255, 255, 255])
			else:
				return utils.get_pixels_avg()[idx - 1 - 70]
			
	def add_pad(self, patch, idx, patch_size, mask = False, Test = False):
	
		original_height, original_width, _ = patch.shape
		assert (original_height, original_width) != (patch_size, patch_size), 'no need to call add_patch for this patch'
		padding_color = self.get_background_pad(idx, Test = Test)
		pad_height, pad_width = patch_size - original_height, patch_size - original_width
		patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
		if mask: return patch
		else:
			for channel in range(3):
				patch[:,:,channel][patch[:,:,channel] == 0] = padding_color[channel]
			return patch


class Slides(): 

	def __init__(self, path_slide): 
	
		self.path_slide = path_slide 
		
	def get_thumbnail(self, level): 
	
		if not hasattr(self, 'slide'):
			self.slide = open_slide(self.path_slide)
		return np.array(self.slide.get_thumbnail(size = self.slide.level_dimensions[level]))
		
	def get_patch(self, level, tile_size, row, col, overlap = 0):
	
		if not hasattr(self, 'slide'):
			self.slide = open_slide(self.path_slide)
		
		if not hasattr(self, 'tiles'): self.tiles = DeepZoomGenerator(self.slide, tile_size = tile_size, overlap = overlap, limit_bounds = False) #no need to save this because it is really fast so it does not matter to do it 50 000 times (but I do it)
		return np.array(self.tiles.get_tile(len(self.tiles.level_tiles) -1 - level, (col, row)))
	
	def get_region(self, location, level, size, type = 'array'): #location is (row, col)
		h, w = location[0] * (2 ** level), location[1] * (2 ** level)
		h, w = int(h), int(w)
		if isinstance(size, int): size = (size, size)
		else:
			height, width = size
			size = (width, height)
		if not hasattr(self, 'slide'):
			self.slide = open_slide(self.path_slide)
		if type == 'array': return np.array(self.slide.read_region((w, h), level, size))[:,:,:3]
		
		else: return self.slide.read_region((w, h), level, size).convert('RGB')
		
	def get_dim(self, level, patch_size, string, overlap = 0): 
	
		assert string in ['THUMB', 'PATCH'], 'error in string argument'
		if not hasattr(self, 'slide'):
			self.slide = open_slide(self.path_slide)
		if string == 'THUMB': return self.slide.level_dimensions[level] #cols, rows
		if string == 'PATCH': 
			if not hasattr(self, 'tiles'): self.tiles = DeepZoomGenerator(self.slide, tile_size = patch_size, overlap = overlap, limit_bounds = False)
		return self.tiles.level_tiles[len(self.tiles.level_tiles) -1 - level]
	
	def get_tissuemask(self, level, Test=False):
	#at a certain level, do directly here the resized tissuemask 
		assert level <= 5, f'code the downscaling of the tissuemask for the level {level}'
		idx = self.path_slide.split('/')[-1].split('.')[0]
		path = f'/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask/{idx}_tissue_mask.png'
		if Test:
			path = f'/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask_test/{idx}_tissue_mask.png'
		tissuemask = cv.imread(path)
		if level == 5: return tissuemask
		factor = pow(2, 5 - level)
		new_height, new_width = factor * tissuemask.shape[0], factor * tissuemask.shape[1]
		resized_tissuemask = cv.resize(tissuemask, (new_width, new_height), interpolation=cv.INTER_LINEAR)
		return resized_tissuemask
		
		
		
		
class Tumors(Slides): 

	def __init__(self, path_slide, path_mask): 
	
		self.path_slide = path_slide 
		self.path_mask = path_mask

	def get_mask(self, level):
		
		if not hasattr(self, 'mask'):
			self.mask = open_slide(self.path_mask)
		mask = np.array(self.mask.get_thumbnail(size = self.mask.level_dimensions[level]))
		if np.max(mask) == 255:
			mask = mask // 255
		return mask
		
	def get_patch(self, level, tile_size, row, col):
	
		if not hasattr(self, 'slide'): self.slide = open_slide(self.path_slide)
		if not hasattr(self, 'tiles'): self.tiles = DeepZoomGenerator(self.slide, tile_size = tile_size, overlap = 0, limit_bounds = False)
		if not hasattr(self, 'mask'): self.mask = open_slide(self.path_mask)
		if not hasattr(self, 'tiles_mask'): self.tiles_mask = DeepZoomGenerator(self.mask, tile_size = tile_size, overlap = 0, limit_bounds = False)
		
		return np.array(self.tiles.get_tile(len(self.tiles.level_tiles) -1 - level, (col, row))), np.array(self.tiles_mask.get_tile(len(self.tiles.level_tiles) -1 - level, (col, row)))
		
	def get_region(self, location, level, size, type = 'array'): #location is (row, col)
	
		h, w = location[0] * (2 ** level), location[1] * (2 ** level)
		h, w = int(h), int(w)
		if isinstance(size, int): size = (size, size)
		else:
			height, width = size
			size = (width, height)
		if not hasattr(self, 'slide'):
			self.slide = open_slide(self.path_slide)
		if not hasattr(self, 'mask'):
			self.mask = open_slide(self.path_mask)
		if type == 'array': return np.array(self.slide.read_region((w, h), level, size))[:,:,:3], np.array(self.mask.read_region((w, h), level, size))[:,:,:3]
		
		else: return self.slide.read_region((w, h), level, size).convert('RGB'), self.mask.read_region((w, h), level, size).convert('RGB')

		
class Tests(Slides):

	def __init__(self, path_slide, path_mask): 
	
		self.path_slide = path_slide 
		self.path_mask = path_mask			
		
		
		
		
		
		

