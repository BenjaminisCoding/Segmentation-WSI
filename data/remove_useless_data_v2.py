import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2 as cv
#import linecache
#import random
import math
from skimage import io 
'''
from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator
'''

import data_utils 

def get_otsumask(path_to_slide, level = 3): 

	slide = open_slide(path_to_slide)
	thumb = slide.get_thumbnail(slide.level_dimensions[level])
	thumb_gray = cv.cvtColor(np.array(thumb), cv.COLOR_BGR2GRAY)
	blur = cv.GaussianBlur(thumb_gray,(5,5),0)
	values = blur.reshape(-1)
	threshold, _ = cv.threshold(values[values < 255], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	_, otsu_mask = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
	return otsu_mask
	
	
def write_relevant(path_to_tumor, path_to_normal, L_mask, tumor = True, Normal = False, level = 2, size = 256, overlap = 0): 
	
	with open("/local/temporary/patches_lv2/images_relevant.txt", "w") as f:
	#with open("/local/temporary/patches_lv2/images_relevant.txt", "a") as f:
	
		if Normal: 
		
			return #code to be written when required to 
			
		if tumor:
		
			for idx in L_mask: 
			
				print(idx, 'tumor')
				name = 'tumor_'
				name_mask = 'tumor_'
				if idx < 10:
					name = name + f'00{idx}' 
					name_mask = name_mask + f'00{idx}_Mask'
				elif 10 <= idx < 100: 
					name = name + f'0{idx}'
					name_mask = name_mask + f'0{idx}_Mask'
				else: 
					name = name + f'{idx}'
					name_mask = name_mask + f'{idx}_Mask'
				root_dir = os.path.join(path_to_tumor, name + '.tif')
				otsu_mask = get_otsumask(root_dir)
				print(type(otsu_mask), 'otsu mask type')
				cols, rows, size_patch_real = get_dim(otsu_mask.shape, level_otsu = 3, level = level, size = size)
				area = size_patch_real * size_patch_real
				for row in range(rows - 1): 
					for col in range(cols - 1):
						patch = otsu_mask[np.int32(size_patch_real * row) : np.int32(size_patch_real * (row + 1)), np.int32(size_patch_real * col) : np.int32(size_patch_real * (col + 1))]
						patch[patch == 255] = 1 #fastest way to do it probably 
						if np.min(patch) < 1 : #not all white 
							per = (area - np.sum(patch)) / area 
							per = int(per * 10000) / 100
							f.write(name + f'_{row}_{col};{per}\n')


def write_relevant_2(path_to_dir, path_to_tissuemask, tumor_names, tumor = True, Normal = False, level = 2, size = 256, overlap = 0, epsilon = 0.005): #this time using the tissue mask obtained not by using otsu thresholding
	
	with open("/local/temporary/patches_lv2/images_Vhe_relevant.txt", "a") as f:
		with open("/local/temporary/patches_lv2/all_images.txt", "r") as G:
			line_idx, line_idx_temp = -1, -1
			lines = G.readlines()
			if tumor:
				for name in tumor_names: 
					print(name)
					path, path_conv = os.path.join(path_to_tissuemask, name + '_tissue_mask.png'), os.path.join(path_to_tissuemask, name + '_tissue_mask_conv.png')
					tissuemask, tissuemask_conv = io.imread(path), io.imread(path_conv)
					cols, rows, size_patch_real = get_dim(tissuemask.shape, level_otsu = 5, level = level, size = size)
					area = size_patch_real ** 2 
					for row in range(rows - 1):
						for col in range(cols - 1):
							patch = tissuemask[np.int32(size_patch_real * row) : np.int32(size_patch_real * (row + 1)), np.int32(size_patch_real * col) : np.int32(size_patch_real * (col + 1))]
							patch_conv = tissuemask_conv[np.int32(size_patch_real * row) : np.int32(size_patch_real * (row + 1)), np.int32(size_patch_real * col) : np.int32(size_patch_real * (col + 1))]
							patch[patch == 255] = 1
							patch_conv[patch_conv == 255] = 1
							per = np.sum(patch) / area 
							if per > 0.005 and np.mean(patch_conv) >= 0.1: 
								per = int(per * 10000) / 100
								name_patch = name + f'_{row}_{col}'
								for line in lines[line_idx + 1:]:
									line_idx_temp += 1
									if line.split(';')[0] == name_patch: 
										to_be_written = line.split('\n')[0] + f';{per}\n'
										f.write(to_be_written)
										line_idx = line_idx_temp
										break 

def add_pos(): #because I forgot to add all the positiv patches, and this create errors 
	with open("/local/temporary/patches_lv2/images_pos.txt", "r") as G:		
		lines_pos = G.readlines()
	with open("/local/temporary/patches_lv2/images_Vhe_relevant.txt", "r") as f:	
		lines = f.readlines()	
	with open("/local/temporary/patches_lv2/images_Vhe_relevant.txt", "a") as f:	
		for line in lines_pos:
			if not line in lines:
				f.write(line)
				print(line)
	return 					
										

def get_dim(otsu_mask_shape, level_otsu, level, size): 

	size_real = size / (2 ** (level_otsu - level))
	rows, cols = math.ceil(otsu_mask_shape[0] / size_real),  math.ceil(otsu_mask_shape[1] / size_real)
	return cols, rows, size_real  

	
def com_size(path_to_dir, L_mask) : 
	dic = {}
	for idx in L_mask: 
		if idx < 10:
			num = f'00{idx}' 

		elif 10 <= idx < 100: 
			num = f'0{idx}'

		else: 
			num = f'{idx}'
			
		with open(os.path.join(path_to_dir, 'all_images.txt'), "r") as f:
			i = 0 
			for line in f: 
				img_id = line.split(";")[0]
				number = img_id.split('_')[1]
				if number == num: 
					i += 1
			with open("/local/temporary/patches_lv2/images_relevant.txt", "r") as q:
				j = 0
				for line in q: 
					img_id = line.split(";")[0]
					number = img_id.split('_')[1]
					if number == num: 
						j += 1	
				dic[num] = [j,i]
	
	return dic 
	
def txt_to_usualtype(): 
	
	filename = os.path.join('/local/temporary/patches_lv2','all_images.txt')
	filename_relevant =  os.path.join('/local/temporary/patches_lv2','images_relevant.txt')
	dic = {} 
	with open(filename_relevant, 'r') as f:
		for line in f:
			name, per = line.split(';')[0], line.split(';')[1]
			dic[name] = [per]
	with open(filename, 'r') as q:
		for line in q:
			name, a1, a2 = line.split(';')[0], line.split(';')[1], line.split(';')[2].split('\n')[0]
			if name in dic.keys(): 
				dic[name].append(a2)
				dic[name].append(a1)
	filename_new = 	os.path.join('/local/temporary/patches_lv2','images_relevant_goodformat.txt')
	with open(filename_new, 'w') as k: 
		for name in dic.keys(): 
			line = name + ';' + dic[name][-1] + ';' + dic[name][-2] + ';' + dic[name][-3] 
			k.write(line)
	return 
		



if __name__ == "__main__":

	path_to_tumor = '/datagrid/Medical/microscopy/CAMELYON16/training/tumor'
	path_to_mask = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask'
	path_to_normal = '/datagrid/Medical/microscopy/CAMELYON16/training/normal'
	
	L_mask = [2, 4, 5, 8, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 33, 34, 39, 44, 48, 52, 55, 58, 59, 60, 61, 64, 69, 71, 72, 73, 75, 76, 77, 79, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
	#L_mask = [2]
	tumor_names = data_utils.get_names_tumors(L_mask)
	path_to_dir = '/local/temporary/patches_lv2'
	path_to_tissuemask = '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask'
	#write_relevant(path_to_tumor, path_to_normal, L_mask, tumor = True, Normal = False, level = 2, size = 256, overlap = 0)
	#write_relevant_2(path_to_dir, path_to_tissuemask, tumor_names, tumor = True, Normal = False, level = 2, size = 256, overlap = 0, epsilon = 0.005)
	#print(com_size('/local/temporary/patches_lv2', L_mask))
	#txt_to_usualtype()
	add_pos()

	 
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
