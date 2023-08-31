import os 
from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator
#from PIL import Image
#from skimage import io 
import data_utils 
from visu_slides import Files 
import cv2 as cv
import numpy as np
import random
from tqdm import tqdm 

'''
import sys
sys.path.append('/mnt/home.stud/laposben/Documents/Scripts/data')
from Pipeline_Training import Build_Dataset
dataset = Build_Dataset(level = 2, threshold = 0.25, size_patch = 512)
dataset.create_dataset()
'''
class Build_Dataset():

	def __init__(self, level, threshold, size_patch): 
	
		self.level, self.threshold, self.size_patch = level, threshold, size_patch
	
	def __arch__(self): 
		
		level, size_patch = self.level, self.size_patch 
		path = '/datagrid/personal/laposben/f_segm/dataset'
		if not(os.path.exists(os.path.join(path, str(level)))):
			os.mkdir(os.path.join(path, str(level)))
		if not(os.path.exists(os.path.join(path, str(level), str(size_patch)))):
			os.mkdir(os.path.join(path, str(level), str(size_patch)))
	
	def create_dataset(self):
		self.__arch__()
		path, level, size_patch = '/datagrid/personal/laposben/f_segm/dataset', self.level, self.size_patch
		if os.path.exists(os.path.join(path, str(level), str(size_patch), 'all_images.txt')):
			print('all_images.txt already exists')
			return
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'all_images.txt'), 'w') as A:
			with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'images_pos.txt'), 'w') as P:
				
				f = Files()
				tumors, usable_tumors, non_usable_tumors = data_utils.get_tumor_names([], All = True), data_utils.get_tumor_names([], Usable_only = True), data_utils.get_non_usable_tumors()
				for tumor in tqdm(tumors):
					print(tumor)
					path_tissuemask, path_tumor, path_mask = f.get_path_tissue_mask(tumor), f.get_path_tumor(tumor), f.get_path_mask(tumor)
					tissuemask = cv.imread(path_tissuemask, cv.IMREAD_UNCHANGED)
					assert isinstance(tissuemask, np.ndarray), f'wrong tissue mask path: {path_tissuemask},{type(tissuemask)}'
					tissuemask[tissuemask == 255] = 1
					tumor_slide, mask_slide = open_slide(path_tumor), open_slide(path_mask)
					tumor_tiles, mask_tiles = DeepZoomGenerator(tumor_slide, tile_size = self.size_patch, overlap = 0, limit_bounds = False), DeepZoomGenerator(mask_slide, tile_size = self.size_patch, overlap = 0, limit_bounds = False)
					rows, cols, size_patch_real = data_utils.get_dim(tissuemask.shape, lv_thumb = 5, lv = self.level, size = size_patch)
					area_real, area = size_patch_real ** 2, size_patch ** 2 
					for row in range(rows):
						for col in range(cols):
							
							patch_tissuemask = tissuemask[np.int32(size_patch_real * row) : np.int32(size_patch_real * (row + 1)), np.int32(size_patch_real * col) : np.int32(size_patch_real * (col + 1))]
							mask = np.array(mask_tiles.get_tile(len(mask_tiles.level_tiles) -1 - self.level, (col, row)))[:,:,0]
							mask[mask == 255] = 1
							fill, pos = np.sum(patch_tissuemask) / area_real, np.sum(mask) / area #the area can be less in the borders. Should not be a problem
							fill, pos = int(fill * 1000) / 1000, int(pos * 1000000) / 1000000
							if tumor in usable_tumors:	
								if pos > 0 or fill >= self.threshold:
									line = f'{tumor}_{row}_{col};{pos};{fill}\n'
									A.write(line)
									if pos > 0: 
										P.write(line)
							if tumor in non_usable_tumors:
								if pos > 0: 
									line = f'{tumor}_{row}_{col};{pos};{fill}\n'
									A.write(line)
									P.write(line)
	#shift + tab to remove tab on several lines 							
	def create_big_dataset(self): #function created to accelerate the creation of the dataset because level 0 took too long, but in the end did not used it
	
		self.__arch__()
		path, level, size_patch = '/datagrid/personal/laposben/f_segm/dataset', self.level, self.size_patch
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'images_pos.txt'), 'r') as P:
			lines_pos = P.readlines()
			n_pos = len(lines_pos)
			idx_slide = 0
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'all_images.txt'), 'w') as A:
			f = Files()
			tumors, usable_tumors, non_usable_tumors = data_utils.get_tumor_names([], All = True), data_utils.get_tumor_names([], Usable_only = True), data_utils.get_non_usable_tumors()
			limit = 200000 / len(usable_tumors) 
			for tumor in tqdm(tumors):
				if data_utils.str_to_idx(tumor) == 38:
					continue
				print(tumor)
				if tumor in usable_tumors:
					path_tissuemask, path_tumor, path_mask = f.get_path_tissue_mask(tumor), f.get_path_tumor(tumor), f.get_path_mask(tumor)
					tissuemask = cv.imread(path_tissuemask, cv.IMREAD_UNCHANGED)
					assert isinstance(tissuemask, np.ndarray), f'wrong tissue mask path: {path_tissuemask},{type(tissuemask)}'
					tissuemask[tissuemask == 255] = 1
					tumor_slide, mask_slide = open_slide(path_tumor), open_slide(path_mask)
					tumor_tiles, mask_tiles = DeepZoomGenerator(tumor_slide, tile_size = self.size_patch, overlap = 0, limit_bounds = False), DeepZoomGenerator(mask_slide, tile_size = self.size_patch, overlap = 0, limit_bounds = False)
					rows, cols, size_patch_real = data_utils.get_dim(tissuemask.shape, lv_thumb = 5, lv = self.level, size = size_patch)
					area_real, area = size_patch_real ** 2, size_patch ** 2 
					n_lines = 0 
					idx_rows, idx_cols = [k for k in range(rows)], [k for k in range(cols)]
					chosen = []
					iters = 0
					while n_lines < limit:
						row, col = random.choice(idx_rows), random.choice(idx_cols)
						iters += 1 
						if iters > 200000:
							break
						if not (row, col) in chosen:
							chosen.append((row, col))
							patch_tissuemask = tissuemask[np.int32(size_patch_real * row) : np.int32(size_patch_real * (row + 1)), np.int32(size_patch_real * col) : np.int32(size_patch_real * (col + 1))]
							mask = np.array(mask_tiles.get_tile(len(mask_tiles.level_tiles) -1 - self.level, (col, row)))[:,:,0]
							mask[mask == 255] = 1
							fill, pos = np.sum(patch_tissuemask) / area_real, np.sum(mask) / area #the area can be less in the borders. Should not be a problem
							fill, pos = int(fill * 1000) / 1000, int(pos * 1000000) / 1000000
							if pos == 0 and fill >= self.threshold:
								line = f'{tumor}_{row}_{col};{pos};{fill}\n'
								A.write(line)
								n_lines += 1	
								
				idx_tum = data_utils.str_to_idx(tumor)
				idx_pos = int(lines_pos[idx_slide].split('_')[1])
				while idx_tum == idx_pos:
					A.write(lines_pos[idx_slide])
					idx_slide += 1 
					if idx_slide < n_pos:
						break
					idx_pos = int(lines_pos[idx_slide].split('_')[1])
					
				
							
	def create_custom_dataset(self, num_folder, factor, cut):
		#cut parameter allow to not have a surrepresentation of non relevant patches so it does not influence too much the training. 
		
		path, level, size_patch = '/local/temporary/f_segm/dataset', self.level, self.size_patch
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'images_pos.txt'), 'r') as P:	
			lines_pos = P.readlines()
			num_pos = len(lines_pos)
			idxs_pos = [k for k in range(num_pos)]
			
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'all_images.txt'), 'r') as A:
			lines_above_th, lines_under_th = [], []
			for line in A:
				pos, fill = float(line.split(';')[1]), float(line.split(';')[2].split('\n')[0])		
				if pos == 0: 
					if fill > cut[0]:
						lines_above_th.append(line)
					else:
						lines_under_th.append(line)
			
			idxs_above_th, idx_under_th = [k for k in range(len(lines_above_th))], [k for k in range(len(lines_under_th))]	
			
		folder = {}
		size_folder_pos = int((1 / num_folder) * num_pos)
		size_folder_above_th, size_folder_under_th = int(size_folder_pos * factor * cut[1]), int(size_folder_pos * factor * (1 - cut[1]))
		folder_pos, folder_above_th, folder_under_th = data_utils.get_random_samples(idxs_pos, size_folder_pos, num_folder), data_utils.get_random_samples(idxs_above_th, size_folder_above_th, num_folder), data_utils.get_random_samples(idx_under_th, size_folder_under_th, num_folder)
		
		filename_directory = os.path.join(os.path.join(path, str(level), str(size_patch), 'custom_dataset'), f'cut_{cut}_numpos_{num_pos}_factor_{factor}')
		direc = os.path.join(os.path.join(path, str(level), str(size_patch), 'custom_dataset'))
		if not(os.path.exists(direc)):
			os.mkdir(direc)
		if not(os.path.exists(filename_directory)):
			os.mkdir(filename_directory)
		for folder in range(1, num_folder + 1):
			filename = os.path.join(filename_directory, f'{folder}.txt')
			with open(filename, 'w') as G:
				for idx in folder_pos[folder - 1]:
					G.write(lines_pos[idx])
				for idx in folder_above_th[folder - 1]:
					G.write(lines_above_th[idx])
				for idx in folder_under_th[folder - 1]:
					G.write(lines_under_th[idx])
					
					
	def create_custom_dataset_slide_divided(self, num_folder, factor, cut, slides_split = None):
	#the difference with the previous function is that patches in the training and valid set come from different slides. This seperation might better reproduce our final goal: predict patches from new slides, so we choose to add this modification.
	#Moreover, it allows to visualize the results of the algorithm on whole image after the training, to compare different experiences
 
		#path, level, size_patch = '/local/temporary/f_segm/dataset', self.level, self.size_patch ##################################################3
		path, level, size_patch = '/datagrid/personal/laposben/f_segm/dataset', self.level, self.size_patch
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'images_pos.txt'), 'r') as P:	
			lines_pos = P.readlines()
			num_pos = len(lines_pos)
			idxs_pos = [k for k in range(num_pos)]
			
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'all_images.txt'), 'r') as A:
			lines_above_th, lines_under_th = [], []
			for line in A:
				pos, fill = float(line.split(';')[1]), float(line.split(';')[2].split('\n')[0])		
				if pos == 0: 
					if fill > cut[0]:
						lines_above_th.append(line)
					else:
						lines_under_th.append(line)
			
			idxs_above_th, idx_under_th = [k for k in range(len(lines_above_th))], [k for k in range(len(lines_under_th))]	
			
		dic_slides = data_utils.get_pos_per_slide(level = level, patch_size = size_patch)
		num_pos_test = np.sum(list(dic_slides.values()))
		assert num_pos == num_pos_test, f'Error: issue with the construction of the dictionnary, num_pos is {num_pos} amd num_pos_test is {num_pos_test}'
		if slides_split is None:
			folders_slides = self.__get_slides_split__(dic_slides, num_folder, num_pos)
		else:
			folders_slides = slides_split
		folders = {}
		for folder in range(1, num_folder + 1):
			folders[(folder, 'pos')] = []
			folders[(folder, 'neg_under')] = []
			folders[(folder, 'neg_above')] = []
		with open(os.path.join(os.path.join(path, str(level), str(size_patch)), 'all_images.txt'), 'r') as A:
			for line in A:
				tumor_id = int(line.split(';')[0].split('_')[1])
				pos, fill = float(line.split(';')[1]), float(line.split(';')[2].split('\n')[0])		
				for k in range(0, num_folder):
					if tumor_id in folders_slides[k]:
						if pos > 0:
							folders[(k + 1, 'pos')].append(line)
						if fill > cut[0]:
							folders[(k + 1, 'neg_above')].append(line)
						else:
							folders[(k + 1, 'neg_under')].append(line)
							
		filename_directory = os.path.join(os.path.join(path, str(level), str(size_patch), 'custom_dataset'), f'cut_{cut}_numpos_{num_pos}_factor_{factor}')
		direc = os.path.join(os.path.join(path, str(level), str(size_patch), 'custom_dataset'))
		if not(os.path.exists(direc)):
			os.mkdir(direc)
		if not(os.path.exists(filename_directory)):
			os.mkdir(filename_directory)
		poss = []
		for folder in range(1, num_folder + 1):
			filename = os.path.join(filename_directory, f'{folder}.txt')
			pos_current = len(folders[(folder, 'pos')])
			poss.append(pos_current)
			size_folder_under_th = int(pos_current * factor * (1 - cut[1]))
			size_folder_above_th = int(pos_current * factor) - size_folder_under_th
			with open(filename, 'w') as G:
				G.writelines(folders[(folder, 'pos')])	
				G.writelines(random.sample(folders[(folder, 'neg_above')], min(size_folder_above_th, len(folders[(folder, 'neg_above')]) - 1 )))
				if size_folder_under_th != 0: #because if we choose to not use this cut parameter, I do not need to write anything
					G.writelines(random.sample(folders[(folder, 'neg_under')], min(size_folder_under_th, len(folders[(folder, 'neg_under')]) - 1 )))
			
		return folders_slides, poss
		
	def __get_slides_split__(self, dic_slides, num_folder, num_pos): 
	
		folders_slides, folders_slides_pos = [], []
		for k in range(num_folder):
			folders_slides.append([])
		reste, quotient = num_pos % num_folder, num_pos // num_folder #we do this to be sure to use every positive patch, it is usefull when working on patches of high level because we lack positive patches
		while len(folders_slides_pos) < num_folder:
			if reste > 0:
				folders_slides_pos.append(quotient + 1)
				reste -= 1
			else:
				folders_slides_pos.append(quotient)
		print(folders_slides_pos, 'folders_slides_pos')	
		d1, d2 = data_utils.len_datasets(d1 = True), data_utils.len_datasets(d1 = False)
		L1, L2 = [k for k in range(1, d1 + 1)], [k for k in range(d1 + 1, d1 + d2 - 1)]
		L1.append(111)
		L1_str, L2_str = data_utils.get_tumor_names(L1), data_utils.get_tumor_names(L2)
		dic_d1, dic_d2 = {k: dic_slides[k] for k in L1_str}, {k: dic_slides[k] for k in L2_str}
		keys_d1_sorted, keys_d2_sorted = sorted(dic_d1, key=lambda k: dic_d1[k], reverse=True), sorted(dic_d2, key=lambda k: dic_d2[k], reverse=True)
		L1_sorted, L2_sorted = [data_utils.str_to_idx(string) for string in keys_d1_sorted], [data_utils.str_to_idx(string) for string in keys_d2_sorted]
		while len(L1_sorted) > 0:
			a = min(num_folder, len(L1_sorted))
			random_pick = random.sample([k for k in range(num_folder)], k = a)
			for pick in random_pick:
				folders_slides[pick].append(L1_sorted.pop(0))
		while len(L2_sorted) > 0:
			a = min(num_folder, len(L2_sorted))
			random_pick = random.sample([k for k in range(num_folder)], k = a)
			for pick in random_pick:
				folders_slides[pick].append(L2_sorted.pop(0))
				
		return folders_slides
		
##########################################################################################
### need to create function that creates the pos_{folder}.txt and the neg_{folder}.txt ###
##########################################################################################

def create_pos_neg(level, patch_size, name_dataset, slides_split):

	save_path =  f'/datagrid/personal/laposben/f_segm/dataset/{level}/{patch_size}/custom_dataset/{name_dataset}'
	path_txt = f'/datagrid/personal/laposben/f_segm/dataset/{level}/{patch_size}'
	check = [] #to check if it works 
	with open(os.path.join(path_txt, 'all_images.txt'), 'r') as R:
		lines = R.readlines()
	'''
	with open(os.path.join(path_txt, 'images_pos.txt', 'r')) as R:
		lines_pos = R.readlines()	
	'''
	for folder in range(1,6):
		slides = slides_split[folder - 1]
		tbw_pos = []
		tbw_neg = []
		for line in lines:
			tumor_id, pos = int(line.split('_')[1]), float(line.split(';')[1])
			if tumor_id in slides:
				if pos == 0:
					tbw_neg.append(line)
				if pos > 0:
					tbw_pos.append(line)
		with open(os.path.join(save_path, f'pos_{folder}.txt'), 'w') as W:
			W.writelines(tbw_pos)
		with open(os.path.join(save_path, f'neg_{folder}.txt'), 'w') as W:
			W.writelines(tbw_neg)
		check.append(len(tbw_neg) + len(tbw_pos))
	
	assert np.array(check).sum() == len(lines)
						
		


		
if __name__ == "__main__":

	#dataset = Build_Dataset(level = 5, threshold = 0.1, size_patch = 512) #0.35 pour level 0, 0.3 pour level 1, 0.25 pour 1,2,3,  0.2 pour lv4 et 0.1 pour lv5
	#level=2 threshold=0.1 patch_size=512 cut1=0.4 cut2=0.75 factor=1 num_folder=5 python Pipeline_Training.py
	level, threshold, patch_size, cut1, cut2, factor, num_folder = int(os.environ.get('level')), float(os.environ.get('threshold')), int(os.environ.get('patch_size')), os.environ.get('cut1'), os.environ.get('cut2'), float(os.environ.get('factor')), int(os.environ.get('num_folder'))
	if (level and threshold and patch_size and cut1 and cut2 and factor and num_folder) is not None:
		cut1, cut2 = float(cut1), float(cut2)
		print(f"Level: {level}, threshold: {threshold}, patch_size: {patch_size}, cut1: {cut1}, cut2: {cut2}, factor: {factor}, num_folder: {num_folder}")
	else:
		print(f"value provided issue Level: {level}, threshold: {threshold}, patch_size: {patch_size}, cut1: {cut1}, cut2: {cut2}, factor: {factor}, num_folder: {num_folder}")

	
	dataset = Build_Dataset(level = level, threshold = threshold, size_patch = patch_size) #0.3 pour level 0, 0.3 pour level 1, 0.25 pour 1,2,3,  0.2 pour lv4 et 0.1 pour lv5
	#dataset.create_dataset() #to create the dataset all_iamges.txt and images_pos.txt, with threshold precised in the command terminal 
	
	#level=1 threshold=0.1 patch_size=512 cut1=-1 cut2=1 factor=1 num_folder=5 python Pipeline_Training.py
	
	slides_split = []
	for folder in range(1,6):
		path_txt = os.path.join('/datagrid/personal/laposben/f_segm/dataset/0/512/custom_dataset/cut_[-1.0, 1.0]_numpos_86416_factor_1.0', f'{folder}.txt')
		slides_split.append(data_utils.slides_in_txt(path_txt, remove_non_usable = False))
	
	'''
	slides, poss = dataset.create_custom_dataset_slide_divided(num_folder = num_folder, factor = factor, cut = [cut1, cut2], slides_split = slides_split)
	print(slides, poss)
	for slide in slides:
		print(len(slide))			
						
							
	'''						
	#level=2 threshold=0.1 patch_size=512 cut1=-1 cut2=1 factor=1 num_folder=5 python Pipeline_Training.py						
					
	create_pos_neg(level, patch_size, name_dataset 	= 'cut_[-1.0, 1.0]_numpos_7851_factor_1.0', slides_split = slides_split)						
						
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
							
					
	 
