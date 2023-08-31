''' 
The purpose of this code is to generate patches from some dataset 
'''

import numpy as np
'''
from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator
import matplotlib.pyplot as plt
'''
import os 
import linecache
import random

def generate_patches_fromtif(path_to_tumor, path_to_mask, path_to_normal, L_mask, level = 2, size = 256, overlap = 0): 
	#je vais faire le format que j'avais propose pour level = 2
	
	num_normal = len(os.listdir(path_to_normal)) #160
	num_to_mask = len(os.listdir(path_to_mask))
	num_to_tumor = len(os.listdir(path_to_tumor))
	
	path_to_save = '/local/temporary/patches_lv2/patch'
	path_to_label = '/local/temporary/patches_lv2/label' #suppose to copy after so first we do locally 
	#path_to_save = '/mnt/home.stud/laposben/Documents/Scripts/model/data/patches_lv2/patch'
	#path_to_label = '/mnt/home.stud/laposben/Documents/Scripts/model/data/patches_lv2/label'
	
	with open("/local/temporary/patches_lv2/all_images.txt", "a") as f:
	#with open("/mnt/home.stud/laposben/Documents/Scripts/model/data/patches_lv2/all_images.txt", "w") as f:
		#with open("/mnt/home.stud/laposben/Documents/Scripts/model/data/patches_lv2/images_pos.txt", "w") as p:
		with open("/local/temporary/patches_lv2/images_pos.txt", "a") as p:
		
			'''
			for idx in range(1,num_normal+1): 
				print(idx, 'normal')
				if idx != 86: #86 not in the list of data 
								
					name='Normal_'
					if idx < 10:
						name = name + f'00{idx}' 
					elif 10 <= idx < 100: 
						name = name + f'0{idx}'
					else: 
						name = name + f'{idx}'
					root_dir = os.path.join(path_to_normal, name + '.tif')
					slide = open_slide(root_dir)
					tiles = DeepZoomGenerator(slide, tile_size = size, overlap = overlap, limit_bounds = False)
					cols, rows = tiles.level_tiles[len(tiles.level_tiles) -1 - level]
					for row in range(rows - 1):
						for col in range(cols - 1):
							tile = np.array(tiles.get_tile(len(tiles.level_tiles) -1 - level, (col, row)))
							plt.imsave(path_to_save + '/' + name + f'_{row}_{col}.png', tile, pnginfo={'alpha': None})
							avg_tile = np.mean(tile)
							avg_tile_print = int(avg_tile * 10) / 10
							f.write(name + f'_{row}_{col};0;{avg_tile_print}\n')
							
						 
			'''			
						
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
				root_dir_mask = os.path.join(path_to_mask, name_mask + '.tif')	
				slide = open_slide(root_dir)
				mask = open_slide(root_dir_mask)
				tiles = DeepZoomGenerator(slide, tile_size = size, overlap = overlap, limit_bounds = False)
				tiles_mask = DeepZoomGenerator(mask, tile_size = size, overlap = overlap, limit_bounds = False)
				cols, rows = tiles.level_tiles[len(tiles.level_tiles) -1 - level]
				for row in range(rows - 1):
					for col in range(cols - 1):
						tile = np.array(tiles.get_tile(len(tiles.level_tiles) -1 - level, (col, row)))
						tile_mask = np.array(tiles_mask.get_tile(len(tiles_mask.level_tiles) -1 - level, (col, row)))
						avg = np.mean(tile_mask[:,:,0]/255)
						avg_print = int(avg * 1000000) / 10000
						avg_tile = np.mean(tile)
						avg_tile_print = int(avg_tile * 10) / 10
							
						if avg > 0: #if there is tumerous region in this patch, save the mask in the label folder 
							 #plt.imsave(path_to_save + '/' + name + f'_{row}_{col}.png', tile, pnginfo={'alpha': None})
							 plt.imsave(path_to_save + '/' + name + f'_{row}_{col}.png', tile)
							 #plt.imsave(path_to_label + '/' + name + f'_{row}_{col}.png', tile_mask, pnginfo={'alpha': None})
							 plt.imsave(path_to_label + '/' + name + f'_{row}_{col}.png', tile_mask)
							 p.write(name + f'_{row}_{col};{avg_print};{avg_tile_print}\n')
						else: 
							#plt.imsave(path_to_save + '/' + name + f'_{row}_{col}.png', tile, pnginfo={'alpha': None})
							plt.imsave(path_to_save + '/' + name + f'_{row}_{col}.png', tile)
						f.write(name + f'_{row}_{col};{avg_print};{avg_tile_print}\n')
						
						
					
					
					
	return 
	
def generate_txt_seuil(path_to_dir, seuil, factor, valid, test, size_pos, cut, remove_useless = True): 
	'''
	look randomly for indexs can be a way to obtain N patch from a certain category 
	cut = [seuil, percentage above that th]
	'''
	'''
	from the txt file path_to_dir/all_images.txt create other txt files representing particular images
	'''
	if remove_useless : 
		#filename = os.path.join(path_to_dir,'images_relevant_goodformat.txt')
		filename = os.path.join(path_to_dir,'images_Vhe_relevant.txt')
	else:	
		filename = os.path.join(path_to_dir,'all_images.txt')
	filename_pos = os.path.join(path_to_dir,'images_pos.txt')
	with open(filename_pos, 'r') as q: 
		lines_pos = q.readlines()
		num_pos = len(lines_pos)
	num_pos = int(num_pos * size_pos)
	num_neg = factor * num_pos
	
	#comment proceder, je pense on a la taille de chaque fichier. Ensuite, on tire aleatoirement pos and neg, tjs memes proportions et ensuite on ecrit les lignes. 
	size_tot = num_neg + num_pos 
	size_train_neg, size_valid_neg, size_test_neg = int((1 - valid - test) * num_neg), int(valid * num_neg), int(test * num_neg)
	size_train_pos, size_valid_pos, size_test_pos = int((1 - valid - test) * num_pos), int(valid * num_pos), int(test * num_pos)
	if remove_useless: 
		seuil_name = 'removed-seuil'
	else: 
		seuil_name = 'seuil'
	filename_directory = os.path.join(path_to_dir, 'custom_dataset', f'cut_{cut}_seuil_{seuil}_pos_[{size_train_pos},{size_valid_pos},{size_test_pos}]_neg_[{size_train_neg},{size_valid_neg},{size_test_neg}]')

	if not os.path.exists(filename_directory):
		os.mkdir(filename_directory)
		
	filename_new_train = os.path.join(filename_directory,'train.txt')
	filename_new_valid = os.path.join(filename_directory,'valid.txt')
	filename_new_test = os.path.join(filename_directory,'test.txt')
	
	filenames = [filename_new_train, filename_new_valid, filename_new_test]
	sizes_neg = [size_train_neg, size_valid_neg, size_test_neg]
	
	with open(filename, 'r') as f: 
		lines_tot = f.readlines()
		lines_above_th = []
		lines_above_seuil = []
		for lgn in lines_tot: 
			if float(lgn.split(';')[-1].split('\n')[0]) >= cut[0] and float(lgn.split(';')[1]) == 0: 
				lines_above_th.append(lgn)
			elif float(lgn.split(';')[-1].split('\n')[0]) >= seuil and float(lgn.split(';')[1]) == 0: 
				lines_above_seuil.append(lgn)
		len_th, len_seuil = len(lines_above_th), len(lines_above_seuil)
		idxs_pos = np.arange(0,num_pos)
		idxs_pos_train = random.sample(list(idxs_pos), size_train_pos)
		idxs_pos_valid = random.sample(list(np.setdiff1d(idxs_pos, idxs_pos_train)), size_valid_pos)
		idxs_pos_test = random.sample(list(np.setdiff1d(idxs_pos, idxs_pos_train + idxs_pos_valid)), size_test_pos)
		idxs_pos_3 = [idxs_pos_train, idxs_pos_valid, idxs_pos_test]
		for i in range(3):
		
			lines_to_be_written = []
			print(i, sizes_neg[i], len_th, cut[1])
			idx_neg_th = random.sample([k for k in range(0, len_th)], np.int32(sizes_neg[i] * cut[1]))
			idx_neg_seuil = random.sample([k for k in range(0, len_seuil)], np.int32(sizes_neg[i] * (1 - cut[1])))
			for element in idx_neg_th: 
				lines_to_be_written.append(lines_above_th[element])
			for element in idx_neg_seuil: 
				lines_to_be_written.append(lines_above_seuil[element])
			for element in idxs_pos_3[i]: 
				lines_to_be_written.append(lines_pos[element])
			with open(filenames[i], 'w') as G: 
				G.writelines(lines_to_be_written)
			#update the pool of lines 
			lines_above_th_new = []
			lines_above_seuil_new = []
			for k in range(len_th): 
				if k not in idx_neg_th:
					lines_above_th_new.append(lines_above_th[k])
			for k in range(len_seuil): 
				if k not in idx_neg_seuil:
					lines_above_seuil_new.append(lines_above_seuil[k])		
			lines_above_th, lines_above_seuil = lines_above_th_new, lines_above_seuil_new 
			len_th, len_seuil = len(lines_above_th), len(lines_above_seuil) #update len

	return filename_new_train, filename_new_valid, filename_new_test
	

def check_for_duplicate(txt1, txt2): 

	with open(txt1, 'r') as f1, open(txt2, 'r') as f2:
        	lines1 = set(f1.readlines())
       		lines2 = set(f2.readlines())
        	common_lines = lines1.intersection(lines2)
        	return bool(common_lines), common_lines
        
def dataset_different(root_dir, train, valid, test): 
	 
	path_to_train = os.path.join(root_dir, train)	
	path_to_valid= os.path.join(root_dir, valid)	
	path_to_test = os.path.join(root_dir, test)	
	
	if check_for_duplicate(path_to_train, path_to_valid)[0] or check_for_duplicate(path_to_train, path_to_test)[0]  or check_for_duplicate(path_to_test, path_to_valid)[0]:
		return check_for_duplicate(path_to_train, path_to_valid)[1], check_for_duplicate(path_to_train, path_to_test)[1], check_for_duplicate(path_to_test, path_to_valid)[1]
	
	
if __name__ == "__main__":

	path_to_tumor = '/datagrid/Medical/microscopy/CAMELYON16/training/tumor'
	path_to_mask = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask'
	path_to_normal = '/datagrid/Medical/microscopy/CAMELYON16/training/normal'

	L_mask = [2, 4, 5, 8, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 33, 34, 39, 44, 48, 52, 55, 58, 59, 60, 61, 64, 69, 71, 72, 73, 75, 76, 77, 79, 81, 82, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] 

	#generate_patches_fromtif(path_to_tumor, path_to_mask, path_to_normal, L_mask, level = 2, size = 256, overlap = 0)
	remove_useless = True
	seuil = 20
	factor = 3 
	valid, test = 10/100, 40/100 #en pourcentage 
	path_to_dir = '/local/temporary/patches_lv2'
	size_pos = 1 
	train, valid, test = generate_txt_seuil(path_to_dir, seuil, factor, valid, test, size_pos, cut = [75, 0.80], remove_useless = True)
	sets = dataset_different(path_to_dir, train, valid, test)
	print(sets)
	 
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
