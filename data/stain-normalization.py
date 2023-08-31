#from __future__ import division
from PIL import Image
#from openslide import open_slide
#import openslide
#from openslide.deepzoom import DeepZoomGenerator
import os 
import cv2 as cv
#import linecache
#import random
import math
from skimage import io 
import time 
import sys
sys.path.append('/mnt/home.stud/laposben/Documents/Stain_Normalization-master')
import data_utils
from visu_slides import Files, Slides, Tumors
from tqdm import tqdm 

import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane
#%load_ext autoreload
#%autoreload 2


import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline


def stain_normalization(path_to_tumor, tumor_names, tumor_names_d1, path_to_targets, dic, epsilon, bug): 

	for name in tumor_names[20:]: 	
	
		print(name)
		target = io.imread(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets','target_' + name + '.png'))
		with open(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets', 'otsu_threshold_lv3.txt'), 'r') as F: 
			for line in F:
				line_id, beta = line.split(';')[0], float(line.split(';')[2].split('\n')[0])
				if line_id == name:
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

		with open('/local/temporary/patches_lv2/images_Vhe_relevant.txt','r') as F:
			#with open('/mnt/home.stud/laposben/Documents/DATA/macenko_patch/processed_images.txt', 'w') as G:
			with open('/local/temporary/patches_lv2/processed_images.txt', 'a') as G:
				flag = False 
				i = -1
				for line in F: 
					i += 1
					if i % 1000 == 0:
						print(i)
					name_patch, idx, seuil, pos = line.split(';')[0], line.split(';')[0].split('_')[1], float(line.split(';')[-1].split('\n')[0]), float(line.split(';')[1])
					line_id = 'tumor_' + idx
					if name == line_id and (seuil >= epsilon or pos > 0) and name_patch not in bug: 
						if not(flag): 
							flag = True 
						path_to_save = os.path.join('/local/temporary/patches_lv2/macenko_patch', name_patch + '.png')
						if not(os.path.exists(path_to_save)):
							print(name_patch)
							img = io.imread(os.path.join('/local/temporary/patches_lv2/patch',name_patch + '.png'))[:,:,:3]
							img_processed = n.transform(img, beta)
							im1 = Image.fromarray(img_processed.astype(np.uint8)) 
							im1.save(path_to_save)
							G.write(line)
					if name != line_id and flag and False:
						break
	return
	
def get_vectorsHE_C_2(path_to_tumor, L_mask, path_to_targets, name): 

	stain_matrix_target = None
	target_concentrations = None

			
	path_ = os.path.join(path_to_tumor, name + '.tif')
	slide = open_slide(path_)
	thumb = slide.get_thumbnail(slide.level_dimensions[3])
	thumb_np = np.array(thumb)
	beta = get_beta__(thumb_np)
	target = io.imread(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets','target_' + name + '.png'))
	n = stainNorm_Macenko.Normalizer()
	n.fit(target, beta)
	stain_matrix_target = n.stain_matrix_target
	target_concentrations= n.target_concentrations
			
	return stain_matrix_target, target_concentrations
	
def stain_matrix_avg():
	'''
	function to compute the avg stain_matrix_target accross the dataset d2 of the test set 
	'''
	d2 = data_utils.dataset_test()[1]
	f = Files()
	stain_matrix_target = np.zeros((2,3))
	for idx in tqdm(d2):
		path_test = f.get_path_test(idx)
		test = Slides(path_test)
		thumb = test.get_thumbnail(3)
		beta = get_beta__(thumb)[0]
		target = io.imread(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets','target_' + data_utils.idx_to_str(idx, Test = True) + '.png'))
		n = stainNorm_Macenko.Normalizer()
		n.fit(target, beta)
		stain_matrix_target += n.stain_matrix_target
	stain_matrix_target = stain_matrix_target / len(d2)
	return stain_matrix_target
		
def get_beta(path_to_tumor, tumor_names, path_to_targets): 

	with open(os.path.join(path_to_targets, 'otsu_threshold_lv3.txt'), 'w') as F:
		
		for name in tumor_names: 
			print(name)
			path_ = os.path.join(path_to_tumor, name + '.tif')
			slide = open_slide(path_)
			thumb = slide.get_thumbnail(slide.level_dimensions[3])
			thumb_np = np.array(thumb)
			beta, threshold = get_beta__(thumb_np)	
			beta = int(beta * 100000) / 100000
			threshold = int(threshold)
			F.write(name + f';{threshold};{beta}\n')
			
	return 
	
	
	
def get_target_WSI(path_to_tumor, L_mask, stride, kernel_size): 
	
	target_ = []
	max_ = []
	stain_matrix_target = []
	target_concentrations = []
	for idx in L_mask: 
			
		print(idx, 'tumor')
		name = 'tumor_'
		if idx < 10:
			name = name + f'00{idx}' 
		elif 10 <= idx < 100: 
			name = name + f'0{idx}'
		else: 
			name = name + f'{idx}'
		
		path_ = os.path.join(path_to_tumor, name + '.tif')
		slide = open_slide(path_)
		thumb = slide.get_thumbnail(slide.level_dimensions[3])
		thumb_np = np.array(thumb)
		beta = get_beta__(thumb_np)
		#on va fit notre bail 
		#donc il nous faut la thumbnail, et le beta personnalise 
		#et ensuite partie ou on cherche dans relevant et on transform par bloc de 4*4 patch et on save 
		i, j, maxi = get_target(name, stride, kernel_size)
		target = thumb_np[(i * stride) * 2: (i*stride + kernel_size) * 2, j * stride * 2: (j*stride + kernel_size) * 2] #if i and j are the extremes, there is an error but it is unlikely to happen
		#return target, maxi 
		target_.append(target)
		max_.append(maxi)
		im1 = Image.fromarray(target.astype(np.uint8)) 
		path_target = os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets', f'target_{name}.png')
		with open('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets/targets_max_per.txt', 'a') as F: 
			F.write(f'{name};{maxi}\n')
		im1.save(path_target)
		'''
		n = stainNorm_Macenko.Normalizer()
		n.fit(target, beta)
		stain_matrix_target.append(n.stain_matrix_target)
		target_concentrations.append(n.target_concentrations)
		'''
		
	return target_, max_, stain_matrix_target, target_concentrations

def save_target_test(stride, kernel_size):
	'''
	same as the get_target_WSI but for the test slides because the slides from the testing slides of the dataset2 are quite different from the slides of the training set of the dataset 2
	'''
	target_ = []
	max_ = []
	
	d2 = data_utils.dataset_test()[1]
	f = Files()
	for idx in tqdm(d2):
		idx_str = data_utils.idx_to_str(idx, Test = True)
		print(idx_str)
		path_target = os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets', f'target_{idx_str}.png')
		if os.path.exists(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets', f'target_{idx_str}.png')):
			continue
		path_test = f.get_path_test(idx)
		test = Slides(path_test)
		thumb_np = test.get_thumbnail(3)
		beta = get_beta__(thumb_np)
		i, j, maxi = get_target_test(idx, stride, kernel_size)
		target = thumb_np[(i * stride) * 2: (i*stride + kernel_size) * 2, j * stride * 2: (j*stride + kernel_size) * 2]
		#target_.append(target)
		#max_.append(maxi)
		im1 = Image.fromarray(target.astype(np.uint8)) 
		path_target = os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets', f'target_{idx_str}.png')
		im1.save(path_target)
		print(maxi)
		
	#return target_, max_
		
def get_beta__(thumb_np): 

	thumb_gray = cv.cvtColor(thumb_np, cv.COLOR_BGR2GRAY)
	blur = cv.GaussianBlur(thumb_gray,(5,5),0)
	values = blur.reshape(-1)
	threshold, _ = cv.threshold(values[values < 255], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	#_, otsu_mask = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
	beta = -np.log(threshold / 255)
	return beta, threshold
	
def get_target(name, stride, kernel_size): 
	
	dir_name = '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/'
	threshold = None
	with open(os.path.join(dir_name, 'tumortothreshold.txt'), 'r') as w:
		for line in w:
			id_line = line.split(';')[0]
			if id_line == name:
				threshold = int(line.split(';')[1].split('\n')[0])
				
	otsu_mask = io.imread(os.path.join(dir_name, name + f'_otsu_threshold_{threshold}.png'))
	mask = io.imread(os.path.join(dir_name, name + '_mask.png'))
	otsu_mask_tf1 = remove_tumor(otsu_mask, mask)
	otsu_mask_tf = transform_otsu(otsu_mask)
	otsu_mask_conv, patches = conv_otsu(otsu_mask_tf, stride, kernel_size)
	i, j, maxi = get_best_patch(otsu_mask_conv, kernel_size)
	return i, j, maxi 
	
def get_target_test(idx, stride, kernel_size):
	
	f = Files()
	path_test = f.get_path_test(idx)
	test = Slides(path_test)
	thumb_np = test.get_thumbnail(4)
	thumb_gray = cv.cvtColor(thumb_np, cv.COLOR_BGR2GRAY)
	blur = cv.GaussianBlur(thumb_gray,(5,5),0)
	shapes = blur.shape
	values = blur.reshape(-1)
	threshold, otsu_mask = cv.threshold(values, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	otsu_mask = otsu_mask.reshape(shapes)
	otsu_mask_tf = transform_otsu(otsu_mask)
	otsu_mask_conv, patches = conv_otsu(otsu_mask_tf, stride, kernel_size)
	i, j, maxi = get_best_patch(otsu_mask_conv, kernel_size)
	return i, j, maxi 

def remove_tumor(otsu_mask, mask): 

	otsu_mask_tf = otsu_mask + mask[:,:,0] #mask are 255 where there is the tumor, and otsu_mask are 0 where there are the ROI. mask has 3 channels so only take one  
	return otsu_mask_tf
    
def transform_otsu(otsu_mask): 

    otsu_mask_tf = otsu_mask / 255
    otsu_mask_tf = np.uint8(np.abs(1 - otsu_mask_tf)) ####################################### I seem to extract only void areas, its weird
    return otsu_mask_tf
    
def conv_otsu(otsu_mask, stride = 625, kernel_size = 1050): 

    patches = get_patches(otsu_mask, stride, kernel_size)
    x, y = [], []
    print('patches done')
    for tupl in list(patches.keys()): 
        x.append(tupl[0])
        y.append(tupl[1])
        #print(tupl, 'dic keys')
    rows, cols = np.max(x), np.max(y)
    res = np.zeros((rows + 1, cols + 1))
    for row in tqdm(range(rows + 1)): #####changed here for row in tqdm(range(rows + 1))
        for col in range(cols + 1):  #########for col in range(cols + 1)
            res[row, col] = conv(patches[(row, col)], kernel_size)
            
    return res, patches
    
def get_patches(otsu_mask, stride = 625, kernel_size = 1050): 
    h, w = otsu_mask.shape 
    rows, cols, rrow, rcol = h // stride, w // stride, h % stride, w % stride 
    patches = {}
    for row in range(rows): 
        for col in range(cols): 
            patches[(row, col)] = otsu_mask[stride * row : stride * row + kernel_size, stride * col : stride * col + kernel_size]
    if rrow > 0: 
        for col in range(cols):
            patches[(rows, col)] = otsu_mask[rows * stride:, stride * col : stride * col + kernel_size] 
    if rcol > 0: 
        for row in range(rows):
            patches[(row, cols)] = otsu_mask[stride * row : stride * row + kernel_size, stride * cols:]
    if rrow > 0 and rcol > 0: 
        patches[(rows, cols)] = otsu_mask[rows * stride:, stride * cols:]
            
    return patches

	
def conv(patch, kernel_size): 
    kernel = np.ones((kernel_size, kernel_size))
  
    if patch.shape != (kernel_size, kernel_size): 
        patch_pad = np.zeros((kernel_size, kernel_size))
        patch_pad[:patch.shape[0], :patch.shape[1]] = patch
        return np.sum(patch_pad * kernel)
    
    return np.sum(patch * kernel)
    

    

    
def get_best_patch(otsu_mask_conv, kernel_size): 
	argmax, maxi = np.argmax(otsu_mask_conv), np.max(otsu_mask_conv / (kernel_size ** 2))
	i, j = argmax // otsu_mask_conv.shape[1], argmax % otsu_mask_conv.shape[1]
	return i, j, maxi
    
	
def get_names_tumors(L_mask): 
	'''
	Because I often have to do this operation so to do it once it for all 
	'''
	names = []
	for idx in L_mask: 
			
		name = 'tumor_'
		if idx < 10:
			name = name + f'00{idx}' 
		elif 10 <= idx < 100: 
			name = name + f'0{idx}'
		else: 
			name = name + f'{idx}'

		names.append(name)
	
	return names 
	
def create_array_percentile(seuil1, seuil2):
	a = np.zeros((100,2))
	a[:,0] = np.linspace(0,1,100) * seuil1/0.99 
	a[:,1] = np.linspace(0,1,100) * seuil2/0.99 
	return a
	
	
if __name__ == "__main__":
	
	'''
	path_to_tumor = '/datagrid/Medical/microscopy/CAMELYON16/training/tumor'
	path_to_mask = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask'
	path_to_normal = '/datagrid/Medical/microscopy/CAMELYON16/training/normal'

	L_mask = [2, 4, 5, 8, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 33, 34, 39, 44, 48, 52, 55, 58, 59, 60, 61, 64, 69, 71, 72, 73, 75, 76, 77, 79, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
	#L_mask = [110]
	tumor_names = get_names_tumors(L_mask)
	tumor_names_d1 = tumor_names[:32]
	stride = 105
	kernel_size = 1050
	
	path_to_targets = '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets'

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
	bug = ['tumor_039_173_70', 'tumor_039_187_68', 'tumor_039_188_67', 'tumor_039_189_66', 'tumor_039_190_66', 'tumor_044_153_51', 'tumor_044_156_32', 'tumor_044_157_33']

	start_time = time.time()
	#print(stain_normalization(path_to_tumor, L_mask, stride, kernel_size))
	#stain_vec, stain_c = get_vectorsHE_C(path_to_tumor, L_mask, path_to_targets)
	stain_normalization(path_to_tumor, tumor_names, tumor_names_d1, path_to_targets, dic, epsilon = 15, bug = bug)
	#print(tumor_names)
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Elapsed time: {elapsed_time} seconds")
	'''
	stride = 125
	kernel_size = 1050

	#save_target_test(stride, kernel_size)
	print(stain_matrix_avg())





	
	'''
	#To save the plots
	path_to_targets = '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/targets'
	flag = True
	for idx in L_mask: 
		
		print(idx, 'tumor')
		name = 'tumor_'
		if idx < 10:
			name = name + f'00{idx}' 
		elif 10 <= idx < 100: 
			name = name + f'0{idx}'
		else: 
			name = name + f'{idx}'
			
		
		stain_vec, stain_c = get_vectorsHE_C_2(path_to_tumor, L_mask, path_to_targets, name)

		
		if not(flag): 
			for i in range(8):
			
	   			if i <= 2: 
	    				x[i].append(stain_vec[0,i])

	   			elif i == 3: 
	    				x[i].append(np.max(stain_c[:,0]))
	    				
	   			elif i <= 6: 
	    				x[i].append(stain_vec[1,i - 4])
	    				
	   			elif i == 7: 
	    				x[i].append(np.max(stain_c[:,1]))
	
		
		
		
		if flag: 
			x = []
			for i in range(8):
	   			if i <= 2: 
	    				x.append([stain_vec[0,i]])

	   			elif i == 3: 
	    				x.append([np.max(stain_c[:,0])])
	    				
	   			elif i <= 6: 
	    				x.append([stain_vec[1,i - 4]])
	   
	   			elif i == 7: 
	    				x.append([np.max(stain_c[:,1])])	    				

			flag = False 
	    		
	fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
	for i, ax in enumerate(axs.flatten()):
	    if i <= 2: 
	    	ax.set_ylabel(f'H_{i}')
	    elif i == 3: 
	    	ax.set_ylabel('C_h')
	    elif i <= 6: 
	    	ax.set_ylabel(f'E_{i - 4}')
	    elif i == 7: 
	    	ax.set_ylabel('C_e')
	    y = np.arange(0,len(L_mask))
	    ax.set_xlabel('Image')
	    ax.plot(y, x[i], marker='o', linestyle='None')
	    

	# add a title to the figure
	fig.suptitle('To verify Macenko pre-processing')

	# adjust the spacing between subplots
	fig.tight_layout()

	# show the plot
	plt.savefig(os.path.join(path_to_targets, 'HE_and_C.png'))
	'''

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
