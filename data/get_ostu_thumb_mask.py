import numpy as np
from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator
import matplotlib.pyplot as plt
import os 
import cv2 as cv
#import linecache
#import random
import math
from PIL import Image
import data_utils 
from skimage import io 
import itertools
import sys
sys.path.append('/mnt/home.stud/laposben/Documents/Stain_Normalization-master')
from tqdm import tqdm


import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane

#%load_ext autoreload
#%autoreload 2

from visu_slides import Files, Slides, Tumors


def save_otsu_thumb_mask(path_to_tumor, path_to_mask, L_mask): 
	
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
		
		slide_name = os.path.join(path_to_tumor, name + '.tif')
		mask_name = os.path.join(path_to_mask, name_mask + '.tif')
		slide = open_slide(slide_name)
		try:
			mask = open_slide(mask_name)
		except: 
			print(name, 'not done')
			continue
		thumb = slide.get_thumbnail(slide.level_dimensions[4])
		thumb_mask = mask.get_thumbnail(mask.level_dimensions[4])
		thumb_np = np.array(thumb)
		thumb_mask_np = np.array(thumb_mask)
		thumb_mask_np[thumb_mask_np == 1] = 255
		thumb_gray = cv.cvtColor(thumb_np, cv.COLOR_BGR2GRAY)
		blur = cv.GaussianBlur(thumb_gray,(5,5),0)
		values = blur.reshape(-1)
		threshold, _ = cv.threshold(values[values < 255], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
		_, otsu_mask = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
		thumb_np[thumb_mask_np == 255] = 1
		
		im1, im2, im3 = Image.fromarray(thumb_np.astype(np.uint8)), Image.fromarray(thumb_mask_np.astype(np.uint8)), Image.fromarray(otsu_mask.astype(np.uint8)) 
		path_thumb, path_mask, path_otsu = os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail',f'{name}_thumb.png'), os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail',f'{name}_mask.png'), os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail',f'{name}_otsu_threshold_{int(threshold)}.png')
		if not os.path.exists(path_thumb) or True:
			im1.save(path_thumb)
		else: 
			print(f'{path_thumb} already exists')
		if not os.path.exists(path_mask) or True:
			im2.save(path_mask)
		else: 
			print(f'{path_mask} already exists')
		if not os.path.exists(path_otsu) or True:
			im3.save(path_otsu)
		else: 
			print(f'{path_otsu} already exists')
		
		with open('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tumortothreshold.txt', 'a') as f:
			f.write(f'{name};{int(threshold)}\n')
		'''
		plt.imsave(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail',f'{name}_thumb.png'), thumb_np, alpha=None)
		plt.imsave(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail',f'{name}_mask.png'), thumb_mask_np, alpha=None)
		plt.imsave(os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail',f'{name}_otsu_threshold_{threshold}.png'), otsu_mask, alpha=None)
		'''
		
def save_tissue_mask(path_to_tumor, tumor_names, path_to_save):
	'''
	function that returns the tissue mask. 24/04
	'''
	i = -1
	for name in tumor_names:
		print(name)
		i += 1
		path_saved = os.path.join(path_to_save, name + '_tissue_mask.png')
		if not os.path.exists(path_saved):
			slide_name = os.path.join(path_to_tumor, name + '.tif')
			slide = open_slide(slide_name)
			thumb = np.array(slide.get_thumbnail(slide.level_dimensions[5]))
			#n = stainNorm_Macenko.Normalizer()
			#n.stain_matrix_target = np.array([[0.51477544, 0.69789579, 0.49589606], [0.2688997 , 0.91733805, 0.28823987]])
			if i <= data_utils.len_dataset_d1(All = True):
				stain_matrix = np.array([[0.51477544, 0.69789579, 0.49589606], [0.2688997 , 0.91733805, 0.28823987]])
			else:	
				stain_matrix = np.array([[0.59425265, 0.85084757, 0.33957628], [0.0887658 , 1.08417178, 0.0749678 ]])
				
			res = utils.get_concentrations(thumb, stain_matrix)
			back_I = res.reshape(thumb.shape[0], thumb.shape[1], 2)
			img = back_I[:,:,1]
			img_uint8 = cv.convertScaleAbs(img, alpha=(255.0/np.max(img)))
			#blur = cv.GaussianBlur(img_uint8,(3,3),0)
			ret, binary_img = cv.threshold(img_uint8, 1, 255, cv.THRESH_BINARY)
			im1 = Image.fromarray(binary_img.astype(np.uint8)) 
			im1.save(path_saved)
		
#####################################################################
### here changing the functions to adapt them for the Test slides ###
#####################################################################

def save_thumb_test():
	'''
	function to save thumbnails of the Test slides, to see which Slides belongs to which dataset and to better visualize them
	'''
	test_slides = [k for k in range(1,131)]
	f = Files()
	for idx in tqdm(test_slides):
		path_test = f.get_path_test(idx)
		test = Slides(path_test)
		thumb = test.get_thumbnail(level=5)
		idx_str = data_utils.idx_to_str(idx, Test = True)
		save_path = f'/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/test_thumbs/{idx_str}_thumb.png'
		im = Image.fromarray(thumb.astype(np.uint8))
		im.save(save_path)
		
def save_thumb_normal():
	'''
	function to save thumbnails of the Test slides, to see which Slides belongs to which dataset and to better visualize them
	'''
	normal_slides = [k for k in range(1,161)]
	f = Files()
	for idx in tqdm(normal_slides):
		if idx == 86:
			continue
		path_normal = f.get_path_normal(idx)
		norm = Slides(path_normal)
		thumb = norm.get_thumbnail(level=6)
		idx_str = data_utils.idx_to_str(idx, Test = False, Norm = True)
		save_path = f'/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/normal_thumbs/{idx_str}_thumb.png'
		im = Image.fromarray(thumb.astype(np.uint8))
		im.save(save_path)
		
def save_thumb_testmask():
	'''
	function to save thumbnails of the mask of the Test slides
	'''
	f = Files()
	path = '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/evaluation'
	result_file_list = []
	result_file_list += [each for each in os.listdir(path) if each.endswith('.png')]
	for case in tqdm(result_file_list):
		if case[:4] == 'Test':
			img = np.array(Image.open(os.path.join(path, case)))
			img[img > 0] = 255
			save_path = '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/test_mask/' + case
			im = Image.fromarray(img.astype(np.uint8))
			im.save(save_path)
		
def save_tissue_mask_test():
	'''
	similar function to save_tissue_mask to save the tissue mask of the test slides in the directory /mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask_test
	'''
	
	test_slides = [k for k in range(1,131)]
	f = Files()
	d1, d2 = data_utils.dataset_test()
	for idx in tqdm(test_slides):
		idx_str = data_utils.idx_to_str(idx, Test = True)
		path_save = os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask_test', f'{idx_str}_tissue_mask.png')
		if not os.path.exists(path_save) or idx in d2: ############## idx in d2 to do again the d2
			path_test = f.get_path_test(idx)
			test = Slides(path_test)
			thumb = test.get_thumbnail(level = 5)
			if idx in d1:
				stain_matrix = np.array([[0.51477544, 0.69789579, 0.49589606], [0.2688997 , 0.91733805, 0.28823987]])
			if idx in d2:
				#stain_matrix = np.array([[0.59425265, 0.85084757, 0.33957628], [0.0887658 , 1.08417178, 0.0749678 ]])
				stain_matrix = np.array([[0.56499557, 0.76840242, 0.29207072], [0.1022816,  0.98561009, 0.09998081]])
			
			res = utils.get_concentrations(thumb, stain_matrix)
			back_I = res.reshape(thumb.shape[0], thumb.shape[1], 2)
			img = back_I[:,:,1]
			img_uint8 = cv.convertScaleAbs(img, alpha=(255.0/np.max(img)))
			#blur = cv.GaussianBlur(img_uint8,(3,3),0)
			ret, binary_img = cv.threshold(img_uint8, 1, 255, cv.THRESH_BINARY)
			im1 = Image.fromarray(binary_img.astype(np.uint8)) 
			im1.save(path_save)
			
def save_tissue_mask_HSV():
	'''
	Using here the saturastion method
	'''
	test_slides = [k for k in range(1,131)]
	f = Files()
	for idx in tqdm(test_slides):
		idx_str = data_utils.idx_to_str(idx, Test = True)
		path_save = os.path.join('/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask_test', f'{idx_str}_tissue_mask.png')
		path_test = f.get_path_test(idx)
		test = Slides(path_test)
		thumb = test.get_thumbnail(level = 5)
		hsv_image = cv.cvtColor(thumb, cv.COLOR_BGR2HSV)
		sat = hsv_image[:,:,1]
		_, thresholded = cv.threshold(sat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
		im1 = Image.fromarray(thresholded.astype(np.uint8)) 
		im1.save(path_save)		
		
		
		
			
def test_erode(tumor_names, path_to_save): 
	for name in tumor_names:
		img = io.imread(os.path.join(path_to_save, name + '_tissue_mask.png'))
		kernel = np.ones((3,3), np.uint8)
		img_eroded = cv.erode(img, kernel, iterations = 1)
		img_restored = cv.dilate(img_eroded, kernel, iterations = 1)
		im1 = Image.fromarray(img_restored.astype(np.uint8)) 
		path_saved = os.path.join(path_to_save, name + '_tissue_mask_eroded.png')
		if not os.path.exists(path_saved):
			im1.save(path_saved)
			
def conv(tissuemask, square_size): 

		H, W = tissuemask.shape

		# Calculate the number of squares in each dimension
		num_squares_h = H // square_size
		num_squares_w = W // square_size
		'''
		# Create a view of the image as non-overlapping squares
		square_view = np.lib.stride_tricks.as_strided(
		    tissuemask,
		    shape=(num_squares_h, num_squares_w, square_size, square_size),
		    strides=tissuemask.strides + tissuemask.strides,
		)
		'''
		res_conv = np.zeros((num_squares_h, num_squares_w))
		for row in range(num_squares_h): 
			for col in range(num_squares_w): 
				res_conv[row, col] = np.sum(tissuemask[row * square_size: (row + 1) * square_size, col * square_size: (col + 1) * square_size]) #compute the surface
				#res_conv[col, row] = np.sum(square)
		res_conv = res_conv / (square_size ** 2)
		return res_conv

def get_boolean(arr) : 
	# plus simple si on prend le resultat en entier directement 
	rows, cols = arr.shape  
	boolean = arr > 0.05
	corners = [[0,0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]]
	for row in range(rows): 
		for col in range(cols):
			if [row, col] in corners:
				if  [row, col] in [corners[0]]:
					boolean[row, col] += neighbors(arr[row: row + 2, col: col +2]) 
				if  [row, col] in [corners[1]]:
					boolean[row, col] += neighbors(arr[row - 1: rows, col: col +2]) 
				if  [row, col] in [corners[2]]:
					boolean[row, col] += neighbors(arr[row: row + 2, col - 1: cols]) 
				if  [row, col] in [corners[3]]:
					boolean[row, col] += neighbors(arr[row -1: rows, col - 1 : cols]) 
																			
			elif row == 0: 
				boolean[row, col] += neighbors(arr[row: row + 2, col - 1: col + 2])
			elif row == rows - 1: 
				boolean[row, col] += neighbors(arr[row - 1: rows, col - 1 : col +2])
			elif col == 0: 
				boolean[row, col] += neighbors(arr[row - 1: row + 2, col: col +2])
			elif col == cols - 1: 
				boolean[row, col] += neighbors(arr[row - 1: row + 2, col - 1: cols])
			else: 
				boolean[row, col] += neighbors(arr[row - 1: row + 2, col - 1: col + 1]) 
				 
	boolean.view(bool)	
	borders = [[0,1,2,3]]
	#corners = [[k,j] for k,i in itertools.product(range(4), range(4))]
	for row in range(rows): 
		for col in range(cols):
			if row not in [0,1,2,rows-3, rows -2, rows -1] and col not in [0,1,2,cols-3, cols -2, cols -1]: 
				boolean[row, col] = (np.sum(boolean[row - 3: row + 4, col - 3: col +4])) >= 2 and boolean[row, col]
			if row not in [0,rows - 1] and col not in [0, cols - 1]: 
				boolean[row, col] =  (boolean[row - 1, col] + boolean[row + 1, col] + boolean[row, col - 1] + boolean[row, col + 1]) >= 1 and boolean[row, col] #to remove solitary points 
	return boolean
	
		
def neighbors(arr): 

	#problem here is I do not try to see if the places with high values of surfaces are close to each other 
	boolean = arr > 0.2
	boolean_2 = arr > 0.1
	return boolean.any() or np.sum(boolean_2) >= 2
			
def compute_mask(path_to_tissuemask, tumor_names, square_size = 100): 

	surfaces = {}
	for name in tumor_names: 
		print(name)
		
		tissuemask = (io.imread(os.path.join(path_to_tissuemask, name + '_tissue_mask.png')) / 255).astype(np.uint8)
		convo = conv(tissuemask, square_size)
		boolean = get_boolean(convo)
		surface = np.zeros_like(tissuemask)
		rows, cols = boolean.shape  
		for row in range(rows): 
			for col in range(cols): 
				if boolean[row, col]: 
					surface[row * square_size: (row + 1) * square_size, col * square_size: (col + 1) * square_size] = np.ones((square_size,square_size)) * np.uint8(255)
		surfaces[name] = surface 
		
	return surfaces
		

	
			
if __name__ == "__main__":	
	'''
	path_to_tumor = '/datagrid/Medical/microscopy/CAMELYON16/training/tumor'
	#path_to_mask = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask'
	path_to_mask = '/mnt/home.stud/laposben/Documents/Mask'
	path_to_normal = '/datagrid/Medical/microscopy/CAMELYON16/training/normal'

	#L_mask = [2, 4, 5, 8, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 33, 34, 39, 44, 48, 52, 55, 58, 59, 60, 61, 64, 69, 71, 72, 73, 75, 76, 77, 79, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
	L_mask = [1, 2]
	tumor_names = data_utils.get_tumor_names(L_mask, All = True)
	path_to_save = '/mnt/home.stud/laposben/Documents/DATA/visualize_thumbnail/tissue_mask'
	path_to_tissuemask = path_to_save
	save_tissue_mask(path_to_tumor, tumor_names, path_to_save)
	#L_mask = [k for k in range(1,112)]
	#save_otsu_thumb_mask(path_to_tumor, path_to_mask, L_mask)
	'''
	'''
	dic = compute_mask(path_to_tissuemask, tumor_names, square_size = 50)
	for name in dic.keys(): 
		im1 = Image.fromarray(dic[name].astype(np.uint8)) 
		path_saved = os.path.join(path_to_save, name + '_tissue_mask_conv.png')
		im1.save(path_saved)
	#test_erode(tumor_names, path_to_save)
	'''
	#save_tissue_mask_test()
	#save_thumb_test()
	#save_tissue_mask_HSV()
	save_thumb_testmask()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
