#from __future__ import division
#from PIL import Image
#from openslide import open_slide
#import openslide
#from openslide.deepzoom import DeepZoomGenerator
import os 
import cv2 as cv
#import linecache
import random
import math
#from skimage import io 
import time 
import sys
sys.path.append('/mnt/home.stud/laposben/Documents/Stain_Normalization-master')
#from visu_slides import Files, Slides, Tumors


'''

import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane
#%load_ext autoreload
#%autoreload 2
'''

import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
	
def get_tumor_names(L_mask, All = False, Usable_only = False): 
	'''
	Because I often have to do this operation so to do it once it for all 
	'''
	if All: 
		return get_tumor_names([k for k in range(1,112)])
	if Usable_only:
		L_all = get_tumor_names([k for k in range(1,112)])
		L = []
		non_usable = get_non_usable_tumors()
		for k in range(len(L_all)):
			if not L_all[k] in non_usable: 
				L.append(L_all[k])
		return L
		
	names = []
	for idx in L_mask: 	
		name = 'tumor_'
		if idx < 10: name = name + f'00{idx}' 
		elif 10 <= idx < 100: name = name + f'0{idx}'
		else: name = name + f'{idx}'
		names.append(name)
	return names 
	
def idx_to_str(idx, Test = False, Norm = False):
	if isinstance(idx, str): return idx 
	tens = len(str(idx))
	if not(Test) and not(Norm):
		if tens == 1: return f'tumor_00{idx}'
		if tens == 2: return f'tumor_0{idx}'
		if tens == 3: return f'tumor_{idx}'
	if Test:
		if tens == 1: return f'Test_00{idx}'
		if tens == 2: return f'Test_0{idx}'
		if tens == 3: return f'Test_{idx}'
	if Norm:
		if tens == 1: return f'Normal_00{idx}'
		if tens == 2: return f'Normal_0{idx}'
		if tens == 3: return f'Normal_{idx}'	
	
def str_to_idx(string):

	if isinstance(string, list):
		res = []
		for element in string:
			res.append(int(element.split('_')[1]))
		return res 
	idx_int = string.split('_')[1]
	return int(idx_int)
	
def get_tumor_already(level): 

	path = '/local/temporary/' + f'patches_lv{level}'
	txt = os.path.join(path, 'all_images.txt')
	if not(os.path.exists(txt)):
		return []
	tumor_names_already = []
	with open(txt, 'r') as A:
		for line in A: 
			tumor_id = line.split(';')[0].split('_')[1]
			tumor_name = 'tumor_' + tumor_id
			if not tumor_name in tumor_names_already:
				tumor_names_already.append(tumor_name) 
	return tunor_names_already
	
def get_dim(thumbnail_shape, lv_thumb, lv, size = 256): 
	
	size_real = size / (2 ** (lv_thumb - lv))
	rows, cols = math.ceil(thumbnail_shape[0] / size_real),  math.ceil(thumbnail_shape[1] / size_real)
	return rows, cols, size_real  
	
def Macenko_fit(tumor_name): #tumor_names_d1 to be changed if we incorporate all the tumors
	
	L_mask = [2, 4, 5, 8, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 33, 34, 39, 44, 48, 52, 55, 58, 59, 60, 61, 64, 69, 71, 72, 73, 75, 76, 77, 79, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
	tumor_names = get_names_tumors(L_mask)
	tumor_names_d1 = tumor_names[:32]
	
	dic = {
	'max_stain_d1': np.array([[0.57, 0.75, 0.58], [0.35, 0.96, 0.35]]),
	'min_stain_d1': np.array([[0.47, 0.65, 0.42], [0.17, 0.88, 0.20]]),
	'stain_avg_d1': np.array([[0.51477544, 0.69789579, 0.49589606], [0.2688997 , 0.91733805, 0.28823987]]),
	'max_stain_d2': np.array([[0.58, 0.82, 0.38], [0.18, 1.02, 0.15]]),
	'min_stain_d2': np.array([[0.49, 0.72, 0.22], [-0.05, 0.97, 0.0]]),
	'stain_avg_d2': np.array([[0.59425265, 0.85084757, 0.33957628], [0.0887658 , 1.08417178, 0.0749678 ]]),
	'max_C99_d1': np.array([[3.25, 6.1]]),
	'min_C99_d1': np.array([[2.25, 3.5]]),
	'target_C_99_d1': create_array_percentile(2.614, 5.075),
	'max_C99_d2': np.array([[3.6, 4.8]]),
	'min_C99_d2': np.array([[2.25, 1]]),
	'target_C_99_d2': create_array_percentile(2.973, 3.08)
	}
	
	target = io.imread(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets','target_' + tumor_name + '.png'))
	with open(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets', 'otsu_threshold_lv3.txt'), 'r') as F: 
		for line in F:
			line_id, beta = line.split(';')[0], float(line.split(';')[2].split('\n')[0])
			if line_id == tumor_name:
				break
	n = stainNorm_Macenko.Normalizer()
	n.fit(target, beta)
	vector_stain = n.stain_matrix_target
	vector_c = n.target_concentrations 
	vector_c_99 = np.percentile(vector_c, 99, axis=0).reshape((1, 2))
	if name in tumor_names_d1:
		if not((vector_stain < dic['max_stain_d1']).all()) or not((vector_stain > dic['min_stain_d1']).all()):
			n.stain_matrix_target = dic['stain_avg_d1']

		if not((vector_c_99 < dic['max_C99_d1']).all()) or not((vector_c_99 > dic['min_C99_d1']).all()):
			n.target_concentrations = dic['target_C_99_d1']

	else: 
		if not((vector_stain < dic['max_stain_d2']).all()) or not((vector_stain > dic['min_stain_d2']).all()):
			n.stain_matrix_target = dic['stain_avg_d2']

		if not((vector_c_99 < dic['max_C99_d2']).all()) or not((vector_c_99 > dic['min_C99_d2']).all()):
			n.target_concentrations = dic['target_C_99_d2']
	return n
	
def get_random_samples(L, size_folder, num_folder):
	
	assert len(L) > size_folder, 'the provided list of indexes is too small'
	folder = []
	temp = L[:]
	for k in range(num_folder):
		if len(temp) >= size_folder:
			random_idxs = random.sample(temp, k = size_folder)
			temp = [elem for elem in temp if elem not in random_idxs]
			folder.append(random_idxs)
		else: 
			remain = [elem for elem in L if elem not in temp]
			yes = random.sample(remain, k = size_folder - len(temp))
			union = set(temp) | set(yes)
			folder.append(list(union))
			temp = []
	return folder
	
def len_datasets(d1 = True): #when we used only the available tumors, but now it is more. Need to be updated.
	if d1 : return 70
	else : return 42
	
def get_non_usable_tumors(forma = 'string'): 

	L = ['tumor_015', 'tumor_018', 'tumor_020', 'tumor_029', 'tumor_033', 'tumor_044', 'tumor_046', 'tumor_051', 'tumor_054', 'tumor_055', 'tumor_079', 'tumor_092', 'tumor_095']
	if forma == 'int':
		L_int = []
		for string in L:
			L_int.append(str_to_idx(string))
		return L_int
	return L 

def get_original_tumors(): #tumeurs qui avaient un masque 	
	return [2, 4, 5, 8, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 33, 34, 39, 44, 48, 52, 55, 58, 59, 60, 61, 64, 69, 71, 72, 73, 75, 76, 77, 79, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] 
	
def dataset_test():
	'''
	return two lists, first d1 and second d2 contained the idx type int of the slides 
	[d1], [d2]
	'''
	return [2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 20, 21, 23, 24, 25, 27, 29, 30, 32, 35, 36, 38, 39, 40, 43, 45, 49, 50, 51, 52, 53, 54, 56, 57, 59, 60, 61, 64, 67, 68, 70, 72, 74, 75, 76, 81, 82, 83, 85, 87, 88, 90, 91, 93, 96, 97, 100, 103, 104, 106, 107, 108, 111, 112, 115, 117, 118, 119, 121, 123, 124, 126, 129, 130], [1, 4, 8, 12, 14, 16, 18, 19, 22, 26, 28, 31, 33, 34, 37, 41, 42, 44, 46, 47, 48, 55, 58, 62, 63, 65, 66, 69, 71, 73, 77, 78, 79, 80, 84, 86, 89, 92, 94, 95, 98, 99, 101, 102, 105, 109, 110, 113, 114, 116, 120, 122, 125, 127, 128] 
	
def get_pos_per_slide(level, patch_size): 	#used to create custom datasets in Pipeline_Training.py. Have to be run on cmpgrid-79 to access the files 
						#work only if the all_images.txt file is organised to the ids of the tumors appear increasingly
	#path = os.path.join('/local/temporary/f_segm/dataset', str(level), str(patch_size))	
	path = os.path.join('/datagrid/personal/laposben/f_segm/dataset', str(level), str(patch_size))
	dic = {}
	tumors = get_tumor_names([], All = True)
	with open(os.path.join(path, 'all_images.txt'), 'r') as F:
		lines = F.readlines()
		idx_max = len(lines)
		idx = 0
		for tumor in tumors:

			dic[tumor] = 0
			id_tumor, pos = lines[idx].split(';')[0].split('_')[1], float(lines[idx].split(';')[1])
			tumor_name = f'tumor_{id_tumor}'
			
			while tumor_name == tumor: 
				idx += 1 
				if pos > 0:
					dic[tumor] += 1
				if idx == idx_max:
					return dic
				id_tumor, pos = lines[idx].split(';')[0].split('_')[1], float(lines[idx].split(';')[1])
				tumor_name = f'tumor_{id_tumor}'
				
	return dic 
	
def slides_in_txt(path_txt, remove_non_usable = True):
	list_slides = []
	with open(path_txt, 'r') as F:
		for line in F:
			id_tumor = int(line.split('_')[1])
			if not id_tumor in list_slides:
				list_slides.append(id_tumor)
	if remove_non_usable:
		non_usable_tumors = get_non_usable_tumors(forma='int')
		new_list_slides = []
		for idx in list_slides:
			if idx not in non_usable_tumors:
				new_list_slides.append(idx)
		return sorted(new_list_slides)
	
	return sorted(list_slides)
'''
def add_pad(patch, idx, patch_size, mask = False, Test = False):
	
	f = Files()
	original_height, original_width, _ = patch.shape
	assert (original_height, original_width) != (patch_size, patch_size), 'no need to call add_patch for this patch'
	padding_color = f.get_background_pad(idx, Test = Test)
	pad_height, pad_width = patch_size - original_height, patch_size - original_width
	patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
	if mask: return patch
	else:
		for channel in range(3):
			patch[:,:,channel][patch[:,:,channel] == 0] = padding_color[channel]
		return patch
'''
	
def get_min_dic(dic, keys): 

	minimum = 0
	idxs = keys[0]
	for key in keys:
		real_key = idx_to_str(key)
		if dic[real_key] < minimum:
			minimum = dic[real_key]
			idxs = key 
	return idxs, minimum

def check_for_duplicate(txt1, txt2): 

	with open(txt1, 'r') as f1, open(txt2, 'r') as f2:
        	lines1 = set(f1.readlines())
       		lines2 = set(f2.readlines())
        	common_lines = lines1.intersection(lines2)
        	return bool(common_lines), common_lines
       
def generate_couples(inting):
	couples = []
	for i in range(0, inting + 1):
		for j in range(i + 1, inting + 1):
			if [i, j] not in couples and [j, i] not in couples:
				couples.append([i, j])
	return couples

        	
def dataset_different(root_dir, train, valid, test): 
	 
	path_to_train = os.path.join(root_dir, train)	
	path_to_valid= os.path.join(root_dir, valid)	
	path_to_test = os.path.join(root_dir, test)	
	
	if check_for_duplicate(path_to_train, path_to_valid)[0] or check_for_duplicate(path_to_train, path_to_test)[0]  or check_for_duplicate(path_to_test, path_to_valid)[0]:
		return check_for_duplicate(path_to_train, path_to_valid)[1], check_for_duplicate(path_to_train, path_to_test)[1], check_for_duplicate(path_to_test, path_to_valid)[1]
		
		
def get_pixels_avg(Test = False):
	
	if Test:
		return {
		
			1: np.array([228., 227., 233.]),
 			4: np.array([235., 231., 236.]),
			8: np.array([226., 227., 229.]),
			12: np.array([227., 224., 231.]),
			14: np.array([227., 224., 231.]),
			16: np.array([229., 226., 233.]),
			18: np.array([229., 227., 232.]),
			19: np.array([229., 226., 232.]),
			22: np.array([227., 225., 230.]),
			26: np.array([228., 225., 231.]),
			28: np.array([227., 224., 231.]),
			31: np.array([225., 222., 229.]),
			33: np.array([227., 224., 231.]),
			34: np.array([223., 222., 226.]),
			37: np.array([227., 224., 231.]),
			41: np.array([228., 225., 232.]),
			42: np.array([227., 225., 230.]),
			44: np.array([227., 224., 231.]),
			46: np.array([226., 227., 227.]),
			47: np.array([227., 224., 231.]),
			48: np.array([227., 224., 231.]),
			55: np.array([228., 225., 232.]),
			58: np.array([227., 224., 231.]),
			62: np.array([226., 223., 230.]),
			63: np.array([227., 224., 231.]),
			65: np.array([226., 228., 227.]),
			66: np.array([227., 224., 231.]),
			69: np.array([228., 225., 232.]),
			71: np.array([228., 225., 232.]),
			73: np.array([228., 225., 232.]),
			77: np.array([227., 224., 231.]),
			78: np.array([227., 224., 231.]),
			79: np.array([228., 225., 232.]),
			80: np.array([226., 223., 230.]),
			84: np.array([227., 225., 230.]),
			86: np.array([228., 225., 232.]),
			89: np.array([227., 224., 231.]),
			92: np.array([228., 225., 232.]),
			94: np.array([229., 226., 233.]),
			95: np.array([227., 224., 231.]),
			98: np.array([228., 225., 232.]),
			99: np.array([228., 224., 231.]),
			101: np.array([226., 223., 230.]),
			102: np.array([229., 226., 233.]),
			105: np.array([227., 224., 231.]),
			109: np.array([227., 224., 231.]),
			110: np.array([225., 227., 226.]),
			113: np.array([228., 225., 231.]),
			114: np.array([226., 228., 227.]),
			116: np.array([218., 205., 189.]),
			120: np.array([227., 224., 231.]),
			122: np.array([226., 228., 227.]),
			125: np.array([227., 224., 230.]),
			127: np.array([227., 224., 231.]),
			128: np.array([227., 224., 231.])
			
			}

	else:
		return [
				np.array([238., 233., 239.]),
				np.array([221., 219., 224.]),
				np.array([221., 220., 224.]),
				np.array([236., 234., 239.]),
				np.array([225., 227., 226.]),
				np.array([238., 234., 239.]),
				np.array([225., 227., 224.]),
				np.array([221., 220., 225.]),
				np.array([221., 220., 225.]),
				np.array([221., 218., 224.]),
				np.array([224., 226., 224.]),
				np.array([224., 222., 227.]),
				np.array([224., 222., 227.]),
				np.array([224., 221., 227.]),
				np.array([236., 234., 239.]),
				np.array([223., 218., 224.]),
				np.array([226., 227., 227.]),
				np.array([226., 228., 227.]),
				np.array([225., 227., 226.]),
				np.array([236., 234., 239.]),
				np.array([221., 219., 224.]),
				np.array([225., 227., 226.]),
				np.array([221., 219., 224.]),
				np.array([221., 219., 224.]),
				np.array([224., 223., 228.]),
				np.array([227., 230., 229.]),
				np.array([225., 227., 226.]),
				np.array([235., 233., 237.]),
				np.array([224., 222., 227.]),
				np.array([227., 229., 227.]),
				np.array([221., 219., 224.]),
				np.array([225., 226., 227.]),
				np.array([225., 227., 226.]),
				np.array([221., 219., 224.]),
				np.array([225., 227., 224.]),
				np.array([226., 228., 225.]),
				np.array([228., 230., 229.]),
				np.array([221., 219., 224.]),
				np.array([221., 219., 224.]),
				np.array([237., 234., 237.])
			]
		
def superimpose(img1, img2, alpha):

	assert img1.shape[:2] == img2.shape[:2], 'two images do not have the same dimensions'
	return cv.addWeighted(img1, alpha, img2, 1 - alpha, 0, dtype = cv.CV_8U)
	
import subprocess
def get_cuda_device(): ### code to obtain a cuda device that is not being used, to accelerate performance and avoid computing several algorithms on the same gpu

	result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, universal_newlines=True)
	output = result.stdout.strip().split('\n')
	memory_usage = [int(x) for x in output]
	return np.argmin(memory_usage)
	
####################################
### use in the new data io class ###
####################################

def add_perturbation(location, patch_size, shape):
	
	h, w = location
	H, W = shape
	new_h, height = perturb(h, H, patch_size)
	new_w, width = perturb(w, W, patch_size)
	assert height>0
	assert width>0
	assert h < H
	assert w < W
	return new_h, new_w, height, width
	
def perturb(x, X, patch_size):
	
	if X - x - patch_size >= 0:
		new_x = random.randint(int(x - patch_size / 2), int(x + patch_size / 2))
	else: 
		new_x = random.randint(int(x - patch_size / 2), x)
	length = min(patch_size, X - new_x)
	return new_x, length
	
def mkdir(save_path):
	if not os.path.exists(save_path):
		os.mkdir(save_path)

def os_environ(variable, value, type_var):
	
	assert type_var in ['int', 'float', 'string']
	if type_var == 'int':
		if variable is None:
			return int(value)
		else:
			return int(variable)
	if type_var == 'float':
		if variable is None:
			return float(value)
		else:
			return float(variable)
	if type_var == 'string':
		if variable is None:
			return str(value)
		else:
			return str(variable)
	if type_var == 'boolean':
		if variable is None:
			return value
		else:
			return bool(variable)
			
		
if __name__ == "__main__":

	path = '/local/temporary/f_segm/dataset/5/512/custom_dataset/cut_[0.15, 0.75]_numpos_611_factor_3'
	txts = [path + f'/{k}.txt' for k in range(1,6)]
	couples = generate_couples(4)
	flag = False
	for couple in couples:
		if check_for_duplicate(txts[couple[0]], txts[couple[1]]):
			flag = True
			print(couple)
	print(check_for_duplicate(txts[1], txts[4])[1])	
























	
	
