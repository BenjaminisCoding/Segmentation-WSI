import os
import math
import cv2 as cv
import tifffile
import numpy as np
import torchvision.transforms as T
import torch 
from tqdm import tqdm 
import openslide
from openslide import open_slide 
from openslide.deepzoom import DeepZoomGenerator 
import torchvision.transforms as T
import time 
import segmentation_models_pytorch as smp
import csv 
from NN import SegModel
from Pipeline_Testing import Pipeline_Testing
import sys 
sys.path.append('/mnt/home.stud/laposben/Documents/Scripts/data')
from visu_slides import Files, Slides, Tumors
import data_utils as utils

def get_global_mask(idx, level, patch_size, pl_module, mask = True):
	
	f = Files()
	t = T.ToTensor()
	path_tumor, path_mask = f.get_path_tumor(idx), f.get_path_mask(idx)
	tumor = Tumors(path_tumor, path_mask)
	cols, rows = tumor.get_dim(level, patch_size, string = 'PATCH')
	thumb = tumor.get_thumbnail(level)
	if mask: 
		mask_true = tumor.get_mask(level)[:,:,0]
		mask_true = np.expand_dims(mask_true, axis=2)
		mask_true = np.repeat(mask_true, 3, axis=2)
	global_mask = np.zeros_like(thumb[:,:,0], dtype = np.float32)
	pl_module.eval()
	i = 0
	with torch.no_grad():
		for row in range(rows):
			for col in range(cols):

				patch = tumor.get_patch(level, patch_size, row, col)[0]
				original_height, original_width, _ = patch.shape
				if original_height != patch_size or original_width != patch_size: #add padding if the tile has not the good dimension because it is close to a border
					padding_color = f.get_background_pad(idx)
					pad_height, pad_width = patch_size - original_height, patch_size - original_width
					patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
					for channel in range(3):
						patch[:,:,channel][patch[:,:,channel] == 0] = padding_color[channel]
					
				inputt = t(patch)
				predicted_mask = pl_module(inputt).sigmoid()
				predicted_mask = predicted_mask.squeeze(0).squeeze(0)
				predicted_mask = np.array(predicted_mask)
				global_mask[patch_size * row: patch_size * (row + 1), patch_size * col: patch_size * (col + 1)] = predicted_mask[:original_height, :original_width]
				i+= 1
	print(i)
	global_mask = np.expand_dims(global_mask, axis=2)
	global_mask = np.repeat(global_mask, 3, axis=2)
	out = utils.superimpose(thumb, global_mask, alpha = 0.7)
	if mask:
		goal = 	utils.superimpose(thumb, mask_true, alpha = 0.7)	
		out = [out, goal]	
	return out, global_mask
	
def get_regions(path_tumor, level, patch_size, overlap):

	dic = {}
	tumor = Slides(path_tumor)
	dims = tumor.get_dim(level, patch_size, string = 'THUMB', overlap = overlap)[1], tumor.get_dim(level, patch_size, string = 'THUMB', overlap = overlap)[0]
	step = patch_size - overlap
	for h in range(0, dims[0], step):
		if h + patch_size > dims[0]:
			height = dims[0] - h
		else:
			height = patch_size
			
		for w in range(0, dims[1], step):
			if w + patch_size > dims[1]:
				width = dims[1] - w
			else:
				width = patch_size
			dic[(h,w)] = tumor.get_region(location = (h, w), level = level, size = (width, height), type = 'PIL')
	return dic


def build_global_prediction_naive(dic, shape, patch_size, overlap, mode = 'MAX'):

	assert mode in ['MAX', 'AVG'], f'mode {mode} is not available. Choose either AVG or MAX mode'
	res = np.zeros((shape[0], shape[1],1))
	res_avg = np.zeros_like(res)
	step, p = patch_size - overlap, patch_size
	for key in dic.keys():
		if isinstance(key,str):
			break
		h, w = key 
		if mode == 'MAX':
			if f'{key[0]}_{key[1]}' in dic.keys():
				width, height = dic[f'{key[0]}_{key[1]}']
				res[h: h + p, w: w + p,0] = np.maximum(res[h: h + p, w: w + p,0], dic[key][:height,:width])
			else: 
				res[h: h + p, w: w + p,0] = np.maximum(res[h: h + p, w: w + p,0], dic[key])
		if mode == 'AVG':
			if f'{key[0]}_{key[1]}' in dic.keys():
				width, height = dic[f'{key[0]}_{key[1]}']
				res[h: h + p, w: w + p,0] += dic[key][:height,:width]
			else:
				res[h: h + p, w: w + p,0] += dic[key]
			res_avg[h: h + p, w: w + p,0] += 1
			
	if mode == 'MAX': return res
	if mode == 'AVG': return res / res_avg
			

def global_mask_naive(idx, level, patch_size, overlap, pl_module, mask = True, mode = 'MAX', batch_size = 32, device = 'cuda:0'):
	# naive because we do not optimize the cache memory 
	
	f = Files()
	t = T.ToTensor()
	path_tumor, path_mask = f.get_path_tumor(idx), f.get_path_mask(idx)
	start = time.time()
	dic = get_regions(path_tumor, level, patch_size, overlap)
	end = time.time()
	#print(end - start, f'Time taken for the function get_regions for tumor {idx}')
	num_patch = len(dic.keys())
	current_batch_size, current_key = 0, 0
	batchs = []
	keys = list(dic.keys())
	start = time.time()
	while current_key < num_patch:
		if num_patch - current_key < 32:
			batch_size = num_patch - current_key
		batch = torch.empty((batch_size,3,patch_size, patch_size))
		current_batch_size = 0
		while current_batch_size < batch_size: 
			patch = dic[keys[current_key]]
			width, height = patch.size
			if (width, height) != (patch_size, patch_size):
				patch = np.array(patch)
				patch = f.add_pad(patch, idx, patch_size)
				dic[f'{keys[current_key][0]}_{keys[current_key][1]}'] = width, height #cols, rows
			batch[current_batch_size] = t(patch)
			current_batch_size += 1
			current_key += 1
		batchs.append(batch)
	end = time.time()
	#print(end - start, f'Time taken to build {len(batchs)} batchs')
	pl_module.eval()
	pl_module.to(device)
	start = time.time()
	with torch.no_grad():
		current_key = -1
		for batch in batchs:
			predicted_mask = pl_module(batch.to(device)).sigmoid()
			for prediction in predicted_mask:
				current_key += 1
				dic[keys[current_key]] = np.array(prediction.cpu().squeeze(0))
		assert current_key == num_patch - 1, 'Error'	
	end = time.time()
	print(end - start, f'Time taken by the model prediction for {num_patch} patches')
	tumor = Tumors(path_tumor, path_mask)
	mask_true = tumor.get_mask(level)
	start = time.time()
	mask_prediction = build_global_prediction_naive(dic, mask_true.shape, patch_size, overlap, mode)
	end = time.time()
	#print(end - start, f'Time taken to build the prediction')
	mask_prediction = np.repeat(mask_prediction, 3, axis=2)
	return mask_true, mask_prediction


def valid_step(folder_valid, name_dataset, level, patch_size, overlap, pl_module, mask = True, mode = 'AVG', fu = None): 

	path_txt = f'/local/temporary/f_segm/dataset/{level}/{patch_size}/custom_dataset/{name_dataset}/{folder_valid}.txt'
	if fu is not None:
		path_txt =  f'/datagrid/personal/laposben/f_segm/dataset/{level}/{patch_size}/custom_dataset/{name_dataset}/{folder_valid}.txt'
	list_slides = utils.slides_in_txt(path_txt)
	dic = {}
	n_cuda = utils.get_cuda_device()
	device = f'cuda:{n_cuda}'
	for tumor in tqdm(list_slides):
		dic[tumor] = global_mask_naive(tumor, level, patch_size, overlap, pl_module, mask = True, mode = mode, batch_size = 32, device = device)
		
	torch.cuda.set_device(device)
	torch.cuda.empty_cache() #ze do this because otherwise during the loop in produce csv global, the cache memory remains in the gpu and and start to use other gpu after, or we want one gpu for one task. 
	#By doing so, the memory cache does not go to 0 but goes to 700 instead of the usual 5600 and it free space so it is good
	return dic 
	
def compute_metrics(dic, folder_valid, name_dataset, level, patch_size, overlap, pl_module, threshold):
	
	#dic =  
	metrics = {}
	t = T.ToTensor()
	non_usable_tumors = utils.str_to_idx(utils.get_non_usable_tumors())
	for tumor in tqdm(dic.keys()):
		if tumor in non_usable_tumors:
			metrics[tumor] = 'non_usable'
		else:
			mask, prediction = dic[tumor]
			_, prediction = cv.threshold(prediction, threshold, 1, cv.THRESH_BINARY)
			mask, prediction = mask[:,:,0], prediction[:,:,0]
			mask, prediction = mask.astype(np.int32), prediction.astype(np.int32)
			mask_np, prediction_np = np.expand_dims(mask, axis = 2), np.expand_dims(prediction, axis = 2)
			mask_tensor, prediction_tensor = t(mask_np).unsqueeze(0), t(prediction_np).unsqueeze(0)
			tp, fp, fn, tn = smp.metrics.get_stats(prediction_tensor.long(), mask_tensor.long(), mode="binary")
			'''
			iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
			f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none") #its dice coefficient 
			f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="none")
			accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="none")
			recall = smp.metrics.recall(tp, fp, fn, tn, reduction="none")
			precision = smp.metrics.precision(tp, fp, fn, tn, reduction="none")
			'''
			metrics[tumor] = tp, fp, fn, tn
				
	return metrics
	
def compute_advanced_metrics(metrics):
	
	advanced_metrics = {}
	for key in metrics.keys():
		if isinstance(metrics[key], str):
			advanced_metrics[key] = 'non_usable'
		else:
			tp, fp, fn, tn = metrics[key]
			advanced_metrics[key] = ['iou_score', smp.metrics.iou_score(tp, fp, fn, tn, reduction="none"), 'dice_coeff', smp.metrics.f1_score(tp, fp, fn, tn, reduction="none"), 'recall', smp.metrics.recall(tp, fp, fn, tn, reduction="none"), 'precision', smp.metrics.precision(tp, fp, fn, tn, reduction="none")]
	return advanced_metrics
	
def graph_th_recall_precision(upper_bound, lower_bound, N_points, dic, folder_valid, name_dataset, level, patch_size, overlap, pl_module): #balance between recall and precision. We want the best recall without having too much precision. 
	
	#thresholds = np.linspace(lower_bound, upper_bound, N_points)
	#thresholds = [10 ** (-k) for k in range(0,7)]
	thresholds = [1]
	for k in range(1,7):
		thresholds.append(10 ** (-k))
		thresholds.append(10 ** (-k) *(1+ 1/4))
		thresholds.append(10 ** (-k) *(1+ 2/4))
		thresholds.append(10 ** (-k) *(1+ 3/4))
	thresholds = sorted(thresholds)
	dic_th = {}
	for key in dic.keys():
		dic_th[f'{key}_recall'] = []
		dic_th[f'{key}_precision'] = []
		dic_th[f'{key}_p'] = []
	for th in tqdm(thresholds):
		metrics = compute_metrics(dic, folder_valid, name_dataset, level, patch_size, overlap, pl_module, th)
		for key in dic.keys():
			if isinstance(metrics[key], str):
				dic_th[f'{key}_recall'].append(-1)
				dic_th[f'{key}_precision'].append(-1)
				dic_th[f'{key}_p'].append(-1)
			else:
				tp, fp, fn, tn = metrics[key]
				recall, precision = smp.metrics.recall(tp, fp, fn, tn, reduction="none"), smp.metrics.precision(tp, fp, fn, tn, reduction="none")
				dic_th[f'{key}_recall'].append(recall)
				dic_th[f'{key}_precision'].append(precision)
				dic_th[f'{key}_p'].append(int(tp + fp))
	
	return thresholds, dic_th

	
def produce_csv(prediction, idx, level, th, val_folder, name_exp, patch_size = 512):

	f = Files()
	H, W = prediction.shape[0], prediction.shape[1]
	factor = pow(2, level)
	wsi_name = f.__idx_to_str__(idx)
	csv_file_path = f'/home.guest/laposben/Documents/Scripts/experiments/{level}/{patch_size}/csv'
	if not(os.path.exists(csv_file_path)):
		os.mkdir(csv_file_path)
	csv_file_path = os.path.join(csv_file_path, name_exp)
	if not(os.path.exists(csv_file_path)):
		os.mkdir(csv_file_path)
	csv_file_path = os.path.join(csv_file_path, str(val_folder))
	if not(os.path.exists(csv_file_path)):
		os.mkdir(csv_file_path)
	csv_file_path = os.path.join(csv_file_path, f'{wsi_name}.csv')
	with open(csv_file_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for (h, w), value in np.ndenumerate(prediction):
			if value > th:
				writer.writerow([value, int(w * factor), int(h * factor)])
				
def produce_csv_global(level, patch_size, arch, encoder, name_model, name_dataset, name_exp, overlap, threshold):

	path = os.path.join('/mnt/home.stud/laposben/Documents/Scripts/experiments', str(level), str(patch_size))
	non_usable_tumors = utils.str_to_idx(utils.get_non_usable_tumors())
	for val_folder in tqdm(range(1,6)):

		model = SegModel(arch, "resnet34", in_channels=3, out_classes=1)
		path_weights = os.path.join(path, f'{name_model}:{val_folder}.pt')
		model.load_state_dict(torch.load(path_weights))
		dic = valid_step(val_folder, name_dataset, level, patch_size, overlap, model, mask = True, mode = 'AVG')
		for key in tqdm(dic.keys()):
			if not key in non_usable_tumors:
				produce_csv(prediction = dic[key][1][:,:,0], idx = key, level = level, th = threshold, val_folder = val_folder, name_exp = name_exp)
	
####### Changing the function that produce overall prediction so that under the threshold the results are 0 to fasten the computation

class build_fast_predictions():

	#############################################################################################
	### to compute the csv one at a time to not take too much memory ############################
	### also here, I will save the image so if I have any bug I can more quickly work zith it ###
	#############################################################################################
	
	def save_pred_global(self, valid_folder, name_dataset, level, patch_size, overlap, pl_module, threshold_fill, name_exp, mask = True, mode = 'AVG', device = None, resized = 0):
	
		path_txt =  f'/datagrid/personal/laposben/f_segm/dataset/{level}/{patch_size}/custom_dataset/{name_dataset}/{valid_folder}.txt'
		list_slides = utils.slides_in_txt(path_txt, remove_non_usable = True) #slides_in_txt changed, so the non_usable are not predicted. Gain time and better results because good predictions can be counted as FP because of poor labeling. 
		if resized > 0:
			level_resized = level + int(np.log(resized) / np.log(2))
		else:
			level_resized = level
		if device is None:
			n_cuda = utils.get_cuda_device()
			device = f'cuda:{n_cuda}'
		folder_path_ = os.path.join(f'/datagrid/personal/laposben/Scripts/experiments/{level}/{patch_size}/predictions')
		if not os.path.exists(folder_path_):
			os.mkdir(folder_path_)
		folder_path_ = os.path.join(f'/datagrid/personal/laposben/Scripts/experiments/{level}/{patch_size}/predictions', name_exp)
		if not os.path.exists(folder_path_):
			os.mkdir(folder_path_)
		folder_path = os.path.join(folder_path_, str(resized))
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)       
		for tumor in tqdm(list_slides):
			save_path = os.path.join(folder_path, f'{utils.idx_to_str(tumor)}_pred.tif')
			if os.path.exists(save_path):
				print(f'tumor {tumor} already saved')                
				continue
			print(tumor)            
			mask_true, mask_prediction = self.global_mask(tumor, level, patch_size, overlap, pl_module, mask = True, mode = mode, batch_size = 64, device = device, threshold_fill = threshold_fill, resized = resized)    
			tifffile.imwrite(save_path, mask_prediction)
            
            
	def valid_step(self, folder_valid, name_dataset, level, patch_size, overlap, pl_module, threshold_fill, mask = True, mode = 'AVG', fu = None, resized = 0): 

		#path_txt = f'/local/temporary/f_segm/dataset/{level}/{patch_size}/custom_dataset/{name_dataset}/{folder_valid}.txt'
		#if fu is not None:
		path_txt =  f'/datagrid/personal/laposben/f_segm/dataset/{level}/{patch_size}/custom_dataset/{name_dataset}/{folder_valid}.txt' ###############
		list_slides = utils.slides_in_txt(path_txt)
		dic = {}
		n_cuda = utils.get_cuda_device()
		device = f'cuda:{n_cuda}'
		for tumor in tqdm(list_slides):
			dic[tumor] = self.global_mask(tumor, level, patch_size, overlap, pl_module, mask = True, mode = mode, batch_size = 32, device = device, threshold_fill = threshold_fill, resized = resized)
			
		torch.cuda.set_device(device)
		torch.cuda.empty_cache() #ze do this because otherwise during the loop in produce csv global, the cache memory remains in the gpu and and start to use other gpu after, or we want one gpu for one task. 
		#By doing so, the memory cache does not go to 0 but goes to 700 instead of the usual 5600 and it free space so it is good
		return dic 
	
	def global_mask(self, idx, level, patch_size, overlap, pl_module, mask = True, mode = 'MAX', batch_size = 32, device = 'cuda:0', threshold_fill = -1, resized = 0):
		
		f = Files()
		t = T.ToTensor()
		path_tumor, path_mask = f.get_path_tumor(idx), f.get_path_mask(idx)
		start = time.time()
		dic = self.get_regions(path_tumor, level, patch_size, overlap, threshold_fill)
		end = time.time()
		print(end - start, f'Time taken for the function get_regions for tumor {idx}')
		num_patch = len(dic.keys())
		current_batch_size, current_key = 0, 0
		batchs = []
		keys = list(dic.keys())
		start = time.time()
		while current_key < num_patch:
			if num_patch - current_key < batch_size:
				batch_size = num_patch - current_key
			batch = torch.empty((batch_size,3,patch_size, patch_size))
			current_batch_size = 0
			while current_batch_size < batch_size:
				if num_patch <= current_key:
					print(num_patch, current_key, len(keys), len(batchs), current_batch_size, batch_size)                    
				patch = dic[keys[current_key]]
				width, height = patch.size
				if (width, height) != (patch_size, patch_size):
					patch = np.array(patch)
					patch = f.add_pad(patch, idx, patch_size)
					dic[f'{keys[current_key][0]}_{keys[current_key][1]}'] = width, height #cols, rows
				batch[current_batch_size] = t(patch)
				current_batch_size += 1
				current_key += 1
			batchs.append(batch)
		end = time.time()
		print(end - start, f'Time taken to build {len(batchs)} batchs')
		pl_module.eval()
		pl_module.to(device)
		start = time.time()
		with torch.no_grad():
			current_key = -1
			for batch in batchs:
				predicted_mask = pl_module(batch.to(device)).sigmoid()
				for prediction in predicted_mask:
					current_key += 1
					if resized == 0:
						dic[keys[current_key]] = np.array(prediction.cpu().squeeze(0))
					else:
						image = np.array(prediction.cpu().squeeze(0))
						new_height, new_width = image.shape[0] // resized, image.shape[1] // resized
						assert image.shape[0] / resized == image.shape[0] // resized and image.shape[1] / resized == image.shape[1] // resized, 'Error: there is a problem with the dimension of the resized dimension'
						dic[keys[current_key]] = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
						#if (image > 0.0001).sum() >= 1:
						#	return dic[keys[current_key]], image

			assert current_key == num_patch - 1, 'Error'	
		end = time.time()
		print(end - start, f'Time taken by the model prediction for {num_patch} patches')
		tumor = Tumors(path_tumor, path_mask)
		level_mask_true = level 
		if resized > 0:
			level_mask_true += int(math.log2(resized))
		print('level_mask_true', level_mask_true)        
		mask_true = tumor.get_mask(level_mask_true)
		start = time.time()
		mask_prediction = self.build_global_prediction_naive(dic, mask_true.shape, patch_size, overlap, mode, resized = resized)
		end = time.time()
		print(end - start, f'Time taken to build the prediction')
		mask_prediction = np.repeat(mask_prediction, 3, axis=2)
		return mask_true, mask_prediction
		
	def build_global_prediction_naive(self, dic, shape, patch_size, overlap, mode = 'MAX', resized = 0):
	
		#here in the AVG method, we do not count the patches that had 0 because of their fill attribute. It is a choice that can be challenged.
		assert mode in ['MAX', 'AVG'], f'mode {mode} is not available. Choose either AVG or MAX mode'
		if resized == 0: 
			res = np.zeros((shape[0], shape[1],1))
			step, p = patch_size - overlap, patch_size
		else: 
			res = np.zeros((shape[0], shape[1],1))
			step, p = (patch_size - overlap) // resized, patch_size // resized
		res_avg = np.zeros_like(res)
		for key in dic.keys():
			if isinstance(key,str):
				break
			h, w = key
			if resized > 0:
				h, w = h // resized, w // resized	 
			if mode == 'MAX':
				if f'{key[0]}_{key[1]}' in dic.keys():
					width, height = dic[f'{key[0]}_{key[1]}']
					res[h: h + p, w: w + p,0] = np.maximum(res[h: h + p, w: w + p,0], dic[key][:height,:width])
				else: 
					res[h: h + p, w: w + p,0] = np.maximum(res[h: h + p, w: w + p,0], dic[key])
			if mode == 'AVG':
				if f'{key[0]}_{key[1]}' in dic.keys():
					width, height = dic[f'{key[0]}_{key[1]}']
					if resized > 0:
						width, height = width // resized, height // resized
					res[h: h + p, w: w + p,0] += dic[key][:height,:width]
				else:
					res[h: h + p, w: w + p,0] += dic[key]
				res_avg[h: h + p, w: w + p,0] += 1
				
		if mode == 'MAX': return res
		if mode == 'AVG': 
			res_avg[res_avg == 0] = 1 #because some patches were not seen
			return res / res_avg
	
	def get_regions(self, path_tumor, level, patch_size, overlap, threshold):

		dic = {}
		tumor = Slides(path_tumor)
		dims = tumor.get_dim(level, patch_size, string = 'THUMB', overlap = overlap)[1], tumor.get_dim(level, patch_size, string = 'THUMB', overlap = overlap)[0]
		step = patch_size - overlap
		tissuemask = tumor.get_tissuemask(level)
		assert np.max(tissuemask) == 255, 'Issue with the max value of tissuemask'      
		assert tissuemask.shape[0] == dims[0], f'issue with tissuemask production tissuemask.shape[0] = {tissuemask.shape[0]} dims[0] = {dims[0]}'
		assert tissuemask.shape[1] == dims[1], f'issue with tissuemask production tissuemask.shape[1] = {tissuemask.shape[1]} dims[1] = {dims[1]}'
		for h in range(0, dims[0], step):
			if h + patch_size > dims[0]:
				height = dims[0] - h
			else:
				height = patch_size
				
			for w in range(0, dims[1], step):
				if w + patch_size > dims[1]:
					width = dims[1] - w
				else:
					width = patch_size
				#region = tumor.get_region(location = (h, w), level = level, size = (width, height), type = 'PIL') ###############former line, changed the .get_region method
				region = tumor.get_region(location = (h, w), level = level, size = (height, width), type = 'PIL')
				fill = (tissuemask[h:h+height,w:w+width,0] // 255).sum() / (height * width)
				if fill > threshold: 
					dic[(h,w)] = region
		return dic	
		
	def compute_csv_all(self, name_exp_preds, th, level, name_dataset, name_exp, valid_folder, patch_size, level_pred, erode_iter = 1):
		'''
		Compute all the csv from the tif files inside a particular path, used to do it on a JupyterNotebook inside a for loop, it is to automatise this
		'''
		resized = pow(2, level - level_pred)	
		path_tiff = f'/datagrid/personal/laposben/Scripts/experiments/{level_pred}/{patch_size}/predictions/{name_exp_preds}/{resized}'
		path_txt =  f'/datagrid/personal/laposben/f_segm/dataset/{level_pred}/{patch_size}/custom_dataset/{name_dataset}/{valid_folder}.txt'
		list_slides_usable = utils.slides_in_txt(path_txt, remove_non_usable = True)
		'''
		non_usable_tumors = utils.get_non_usable_tumors(forma='string') #we do this step because during the first tifffiles experiences, I saved non_usable_tumors so it is not use them
		result_file_list = []
		result_file_list += [each for each in os.listdir(path_tiff) if each.endswith('.tif')]
		'''
		for slide in tqdm(list_slides_usable):
		
			prediction, idx = tifffile.imread(os.path.join(path_tiff, utils.idx_to_str(slide, Test = Test) + '_pred.tif')), slide
			self.compute_csv(prediction, th, idx, level, name_exp, valid_folder, patch_size, level_pred, erode_iter)
		
	
	
	
	def compute_csv(self, prediction, th, idx, level, name_exp, val_folder, patch_size, level_pred, erode_iter = 1):
		
		assert len(prediction.shape) == 3, f'prediction should be of shape (p,p,3) and is {prediction.shape}'
		pred = prediction[:,:,0]
		mask_gray = pred > th
		mask_gray = mask_gray.astype(np.uint8)
		dist_max = self.__get_dist_max__(level)
		k = int(dist_max // 10)
		k = k + abs(k % 2 -1)
		kernel = np.ones((k, k), np.uint8)
		mask_eroded = cv.erode(mask_gray, kernel, iterations=erode_iter) ###########################was iterations=1
		contours, _ = cv.findContours(mask_eroded.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		threshold = int(dist_max) - 2 * k - 1
		long_contours = self.__get_long_contours__(contours, threshold)
		tumor_centers, tumor_probabilities = self.get_probs_centers(long_contours, mask_eroded, pred)
		self.__write_csv__(idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp + f';er:{erode_iter}', val_folder, patch_size, th)
		
	def compute_csv_nolongcnt(self, prediction, th, idx, level, name_exp, val_folder, patch_size, level_pred, erode_iter = 1):
		
		assert len(prediction.shape) == 3, f'prediction should be of shape (p,p,3) and is {prediction.shape}'
		pred = prediction[:,:,0]
		mask_gray = pred > th
		mask_gray = mask_gray.astype(np.uint8)
		dist_max = self.__get_dist_max__(level)
		k = int(dist_max // 10)
		k = k + abs(k % 2 -1)
		kernel = np.ones((k, k), np.uint8)
		mask_eroded = cv.erode(mask_gray, kernel, iterations=erode_iter)
		contours, _ = cv.findContours(mask_eroded.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		tumor_centers, tumor_probabilities = self.get_probs_centers(contours, mask_eroded, pred)
		self.__write_csv__(idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp, val_folder, patch_size, th)
		
	def compute_csv_test(self, prediction, th, idx, level, name_exp, val_folder, patch_size, level_pred):
		
		assert len(prediction.shape) == 3, f'prediction should be of shape (p,p,3) and is {prediction.shape}'
		pred = prediction[:,:,0]
		mask_gray = pred > th
		mask_gray = mask_gray.astype(np.uint8)
		contours, _ = cv.findContours(mask_gray.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		tumor_centers, tumor_probabilities = self.get_probs_centers(contours, mask_gray, pred)
		self.__write_csv__(idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp, val_folder, patch_size, th)	
	
	def __get_dist_max__(self, level):
	
		resolution = 0.243
		#####distance = 275 / (resolution * pow(2, level)) #275 car ils expand dans le code evaluation_code
		distance = 200 / (resolution * pow(2, level))
		return distance 
	
	def __get_long_contours__(self, contours, threshold):
		
		long_contours = []
		for cnt in contours:
			rect = cv.minAreaRect(cnt)
			box = cv.boxPoints(rect)
			box = np.int8(box)
			longuest_side = round(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])))
			if longuest_side > threshold:
				long_contours.append(cnt)
		return long_contours
		
	def get_probs_centers(self, contours, mask_eroded, pred, mode = 'RECTANGLE'):
	
		tumor_centers, tumor_probabilities = [], []
		i = 0
		for cnt in contours:
			M = cv.moments(cnt)
			if M['m00'] == 0:
				continue
			i += 1
			center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
			mask_contour = np.zeros_like(mask_eroded)
			cv.drawContours(mask_contour, [cnt], 0, 1, cv.FILLED)
			tumor_centers.append(center)
			tumor_probabilities.append((pred[mask_contour > 0]).sum() / (mask_contour // np.max(mask_contour)).sum())
		return tumor_centers, tumor_probabilities
	
	def __write_csv__(self, idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp, val_folder, patch_size, th):
	
		f = Files()
		factor = pow(2, level)
		wsi_name = f.__idx_to_str__(idx)
		csv_file_path = f'/datagrid/personal/laposben/Scripts/experiments/{level_pred}/{patch_size}/csv'
		resized = str(pow(2, level - level_pred))
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, resized)
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, name_exp)
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, str(val_folder))
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, str(th))
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, f'{wsi_name}.csv')
		with open(csv_file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for i in range(len(tumor_centers)):
				line = [tumor_probabilities[i], int(tumor_centers[i][0] * factor), int(tumor_centers[i][1] * factor)]
				writer.writerow(line)
class Save_Test():

######################################################################
### Class to produce the prediction and the csv of the test slides ###
######################################################################

	def save_pred_global(self, level, patch_size, overlap, pl_module, threshold_fill, name_exp, mask = True, mode = 'AVG', device = None, resized = 0):
	
		if resized > 0:
			level_resized = level + int(np.log(resized) / np.log(2))
		else:
			level_resized = level
		if device is None:
			n_cuda = utils.get_cuda_device()
			device = f'cuda:{n_cuda}'
		else:
			device = f'cuda:{device}'
		folder_path_ = os.path.join(f'/datagrid/personal/laposben/Scripts/experiments/{level}/{patch_size}/predictions')
		utils.mkdir(folder_path_)
		folder_path_ = os.path.join(folder_path_, 'test')
		utils.mkdir(folder_path_)
		folder_path_ = os.path.join(folder_path_, name_exp)
		utils.mkdir(folder_path_)
		folder_path = os.path.join(folder_path_, str(resized))
		utils.mkdir(folder_path) 
		tests = [k for k in range(1,131)]    
		if not os.path.exists(os.path.join(folder_path, 'monitor_time.txt')):
			with open(os.path.join(folder_path, 'monitor_time.txt'), 'w') as W:
				#create the txt file
				pass
		with open(os.path.join(folder_path, 'monitor_time.txt'), 'a') as A:
			for test in tqdm(tests):
				save_path = os.path.join(folder_path, f'{utils.idx_to_str(test, Test = True)}_pred.tif')
				if os.path.exists(save_path):
					print(f'Test {test} already saved')                
					continue
				print(test)
				start = time.time()            
				mask_prediction = self.global_mask(test, level, patch_size, overlap, pl_module, mask = True, mode = mode, batch_size = 64, device = device, threshold_fill = threshold_fill, resized = resized)    
				end = time.time()
				A.write(f'{test}:{end-start}\n')
				tifffile.imwrite(save_path, mask_prediction)	
			
	def global_mask(self, idx, level, patch_size, overlap, pl_module, mask = True, mode = 'MAX', batch_size = 32, device = 'cuda:0', threshold_fill = -1, resized = 0):
		
		f = Files()
		t = T.ToTensor()
		path_test= f.get_path_test(idx)
		start = time.time()
		dic = self.get_regions(path_test, level, patch_size, overlap, threshold_fill)
		end = time.time()
		print(end - start, f'Time taken for the function get_regions for tumor {idx}')
		num_patch = len(dic.keys())
		current_batch_size, current_key = 0, 0
		batchs = []
		keys = list(dic.keys())
		start = time.time()
		while current_key < num_patch:
			if num_patch - current_key < batch_size:
				batch_size = num_patch - current_key
			batch = torch.empty((batch_size,3,patch_size, patch_size))
			current_batch_size = 0
			while current_batch_size < batch_size:
				if num_patch <= current_key:
					print(num_patch, current_key, len(keys), len(batchs), current_batch_size, batch_size)                    
				patch = dic[keys[current_key]]
				width, height = patch.size
				if (width, height) != (patch_size, patch_size):
					patch = np.array(patch)
					patch = f.add_pad(patch, idx, patch_size, Test = True)
					dic[f'{keys[current_key][0]}_{keys[current_key][1]}'] = width, height #cols, rows
				batch[current_batch_size] = t(patch)
				current_batch_size += 1
				current_key += 1
			batchs.append(batch)
		end = time.time()
		print(end - start, f'Time taken to build {len(batchs)} batchs')
		pl_module.eval()
		pl_module.to(device)
		print(f'device is {device} for pred {idx}')
		start = time.time()
		with torch.no_grad():
			current_key = -1
			for batch in batchs:
				predicted_mask = pl_module(batch.to(device)).sigmoid()
				for prediction in predicted_mask:
					current_key += 1
					if resized == 0:
						dic[keys[current_key]] = np.array(prediction.cpu().squeeze(0))
					else:
						image = np.array(prediction.cpu().squeeze(0))
						new_height, new_width = image.shape[0] // resized, image.shape[1] // resized
						assert image.shape[0] / resized == image.shape[0] // resized and image.shape[1] / resized == image.shape[1] // resized, 'Error: there is a problem with the dimension of the resized dimension'
						dic[keys[current_key]] = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
						#if (image > 0.0001).sum() >= 1:
						#	return dic[keys[current_key]], image

			assert current_key == num_patch - 1, 'Error'	
		end = time.time()
		print(end - start, f'Time taken by the model prediction for {num_patch} patches')
		level_mask_true = level 
		if resized > 0:
			level_mask_true += int(math.log2(resized))
		print('level_mask_true', level_mask_true)        
		test = Slides(path_test)
		thumb = test.get_thumbnail(level_mask_true)
		start = time.time()
		mask_prediction = self.build_global_prediction_naive(dic, thumb.shape, patch_size, overlap, mode, resized = resized)
		end = time.time()
		print(end - start, f'Time taken to build the prediction')
		mask_prediction = np.repeat(mask_prediction, 3, axis=2)
		return mask_prediction	
		
	def get_regions(self, path_test, level, patch_size, overlap, threshold):

		dic = {}
		test = Slides(path_test)
		dims = test.get_dim(level, patch_size, string = 'THUMB', overlap = overlap)[1], test.get_dim(level, patch_size, string = 'THUMB', overlap = overlap)[0]
		step = patch_size - overlap
		tissuemask = test.get_tissuemask(level, Test = True)
		assert np.max(tissuemask) == 255, 'Issue with the max value of tissuemask'      
		assert tissuemask.shape[0] == dims[0], f'issue with tissuemask production tissuemask.shape[0] = {tissuemask.shape[0]} dims[0] = {dims[0]}'
		assert tissuemask.shape[1] == dims[1], f'issue with tissuemask production tissuemask.shape[1] = {tissuemask.shape[1]} dims[1] = {dims[1]}'
		for h in range(0, dims[0], step):
			if h + patch_size > dims[0]:
				height = dims[0] - h
			else:
				height = patch_size
				
			for w in range(0, dims[1], step):
				if w + patch_size > dims[1]:
					width = dims[1] - w
				else:
					width = patch_size
				#region = tumor.get_region(location = (h, w), level = level, size = (width, height), type = 'PIL') ###############former line, changed the .get_region method
				region = test.get_region(location = (h, w), level = level, size = (height, width), type = 'PIL')
				fill = (tissuemask[h:h+height,w:w+width,0] // 255).sum() / (height * width)
				if fill > threshold: 
					dic[(h,w)] = region
		return dic
	
	def build_global_prediction_naive(self, dic, shape, patch_size, overlap, mode = 'MAX', resized = 0):
	
		#here in the AVG method, we do not count the patches that had 0 because of their fill attribute. It is a choice that can be challenged.
		assert mode in ['MAX', 'AVG'], f'mode {mode} is not available. Choose either AVG or MAX mode'
		if resized == 0: 
			res = np.zeros((shape[0], shape[1],1))
			step, p = patch_size - overlap, patch_size
		else: 
			res = np.zeros((shape[0], shape[1],1))
			step, p = (patch_size - overlap) // resized, patch_size // resized
		res_avg = np.zeros_like(res)
		for key in dic.keys():
			if isinstance(key,str):
				break
			h, w = key
			if resized > 0:
				h, w = h // resized, w // resized	 
			if mode == 'MAX':
				if f'{key[0]}_{key[1]}' in dic.keys():
					width, height = dic[f'{key[0]}_{key[1]}']
					res[h: h + p, w: w + p,0] = np.maximum(res[h: h + p, w: w + p,0], dic[key][:height,:width])
				else: 
					res[h: h + p, w: w + p,0] = np.maximum(res[h: h + p, w: w + p,0], dic[key])
			if mode == 'AVG':
				if f'{key[0]}_{key[1]}' in dic.keys():
					width, height = dic[f'{key[0]}_{key[1]}']
					if resized > 0:
						width, height = width // resized, height // resized
					res[h: h + p, w: w + p,0] += dic[key][:height,:width]
				else:
					res[h: h + p, w: w + p,0] += dic[key]
				res_avg[h: h + p, w: w + p,0] += 1
				
		if mode == 'MAX': return res
		if mode == 'AVG': 
			res_avg[res_avg == 0] = 1 #because some patches were not seen
			return res / res_avg

	def compute_csv_all(self, name_exp_preds, th, level, name_exp, patch_size, level_pred, erode_iter = 1):
		'''
		Compute all the csv from the tif files inside a particular path, used to do it on a JupyterNotebook inside a for loop, it is to automatise this
		'''
		resized = pow(2, level - level_pred)
		csv_file_path = f'/datagrid/personal/laposben/Scripts/experiments/{level_pred}/{patch_size}/csv/{resized}/{name_exp};er:{erode_iter}/test/{th}'
		path_tiff = f'/datagrid/personal/laposben/Scripts/experiments/{level_pred}/{patch_size}/predictions/test/{name_exp_preds}/{resized}'
		list_slides_usable = [k for k in range(1,131)]
		for slide in tqdm(list_slides_usable):
			print(os.path.join(csv_file_path, utils.idx_to_str(slide, Test = True) + '.csv'))
			if os.path.exists(os.path.join(csv_file_path, utils.idx_to_str(slide, Test = True) + '.csv')):
				print(utils.idx_to_str(slide, Test = True) + '.csv already saved')
				continue
			prediction, idx = tifffile.imread(os.path.join(path_tiff, utils.idx_to_str(slide, Test = True) + '_pred.tif')), slide
			self.compute_csv(prediction, th, idx, level, name_exp, patch_size, level_pred, erode_iter)	
	
	def compute_csv(self, prediction, th, idx, level, name_exp, patch_size, level_pred, erode_iter = 1):
		
		assert len(prediction.shape) == 3, f'prediction should be of shape (p,p,3) and is {prediction.shape}'
		pred = prediction[:,:,0]
		mask_gray = pred > th
		mask_gray = mask_gray.astype(np.uint8)
		dist_max = self.__get_dist_max__(level)
		k = int(dist_max // 10)
		k = k + abs(k % 2 -1)
		kernel = np.ones((k, k), np.uint8)
		mask_eroded = cv.erode(mask_gray, kernel, iterations=erode_iter) ###########################was iterations=1
		contours, _ = cv.findContours(mask_eroded.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		threshold = int(dist_max) - 2 * k - 1
		long_contours = self.__get_long_contours__(contours, threshold)
		tumor_centers, tumor_probabilities = self.get_probs_centers(long_contours, mask_eroded, pred)
		self.__write_csv__(idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp + f';er:{erode_iter}', patch_size, th)
		
	def compute_csv_nolongcnt(self, prediction, th, idx, level, name_exp, patch_size, level_pred, erode_iter = 1):
		
		assert len(prediction.shape) == 3, f'prediction should be of shape (p,p,3) and is {prediction.shape}'
		pred = prediction[:,:,0]
		mask_gray = pred > th
		mask_gray = mask_gray.astype(np.uint8)
		dist_max = self.__get_dist_max__(level)
		k = int(dist_max // 10)
		k = k + abs(k % 2 -1)
		kernel = np.ones((k, k), np.uint8)
		mask_eroded = cv.erode(mask_gray, kernel, iterations=erode_iter)
		contours, _ = cv.findContours(mask_eroded.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		tumor_centers, tumor_probabilities = self.get_probs_centers(contours, mask_eroded, pred)
		self.__write_csv__(idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp, val_folder, patch_size, th)
		
	def compute_csv_test(self, prediction, th, idx, level, name_exp, patch_size, level_pred):
		
		assert len(prediction.shape) == 3, f'prediction should be of shape (p,p,3) and is {prediction.shape}'
		pred = prediction[:,:,0]
		mask_gray = pred > th
		mask_gray = mask_gray.astype(np.uint8)
		contours, _ = cv.findContours(mask_gray.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
		tumor_centers, tumor_probabilities = self.get_probs_centers(contours, mask_gray, pred)
		self.__write_csv__(idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp, val_folder, patch_size, th)	
	
	def __get_dist_max__(self, level):
	
		resolution = 0.243
		#####distance = 275 / (resolution * pow(2, level)) #275 car ils expand dans le code evaluation_code
		distance = 200 / (resolution * pow(2, level))
		return distance 
	
	def __get_long_contours__(self, contours, threshold):
		
		long_contours = []
		for cnt in contours:
			rect = cv.minAreaRect(cnt)
			box = cv.boxPoints(rect)
			box = np.int8(box)
			longuest_side = round(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])))
			if longuest_side > threshold:
				long_contours.append(cnt)
		return long_contours
		
	def get_probs_centers(self, contours, mask_eroded, pred, mode = 'RECTANGLE'):
	
		tumor_centers, tumor_probabilities = [], []
		i = 0
		for cnt in contours:
			M = cv.moments(cnt)
			if M['m00'] == 0:
				continue
			i += 1
			center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
			mask_contour = np.zeros_like(mask_eroded)
			cv.drawContours(mask_contour, [cnt], 0, 1, cv.FILLED)
			tumor_centers.append(center)
			tumor_probabilities.append((pred[mask_contour > 0]).sum() / (mask_contour // np.max(mask_contour)).sum())
		return tumor_centers, tumor_probabilities
	
	def __write_csv__(self, idx, tumor_centers, tumor_probabilities, level, level_pred, name_exp, patch_size, th):
	
		f = Files()
		factor = pow(2, level)
		wsi_name = utils.idx_to_str(idx, Test = True)
		csv_file_path = f'/datagrid/personal/laposben/Scripts/experiments/{level_pred}/{patch_size}/csv'
		resized = str(pow(2, level - level_pred))
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, resized)
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, name_exp)
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, 'test')
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, str(th))
		if not(os.path.exists(csv_file_path)):
			os.mkdir(csv_file_path)
		csv_file_path = os.path.join(csv_file_path, f'{wsi_name}.csv')
		with open(csv_file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for i in range(len(tumor_centers)):
				line = [tumor_probabilities[i], int(tumor_centers[i][0] * factor), int(tumor_centers[i][1] * factor)]
				writer.writerow(line)			
			
if __name__ == "__main__":

	csv_ = os.environ.get('csv_')
	test_ = os.environ.get('test_')
	if csv_ is None:
	
		if test_ is None:
	
			#level=5 name_dataset='cut_[0.2, 0.9]_numpos_611_factor_1.0' valid_folder=1 python test.py
			#level=0 name_dataset='cut_[-1.0, 1.0]_numpos_86416_factor_1.0' encoder=resnet34 valid_folder=1 arch=UNET name_exp=UNETcombineloss device= python test.py
			level, patch_size, arch, encoder, name_model, valid_folder, name_dataset, overlap, threshold, name_exp, device = int(os.environ.get('level')), os.environ.get('patch_size'), os.environ.get('arch'), os.environ.get('encoder'), os.environ.get('name_model'), int(os.environ.get('valid_folder')), os.environ.get('name_dataset'), os.environ.get('overlap'), os.environ.get('threshold'), os.environ.get('name_exp'), os.environ.get('device')
			if patch_size is None: patch_size = 512
			else: patch_size = int(patch_size)
			if overlap is None: overlap = 128
			else: overlap = int(overlap)
			if arch is None: arch = 'FPN'
			if encoder is None: encoder= 'resnet34'
			obj = build_fast_predictions()
			pipe = Pipeline_Testing(level, patch_size, name_dataset, valid_folder, gpu=0)
			model = pipe.create_model(arch = arch, encoder = 'resnet34', save_path = pipe.path, name_model = ',', valid_folder = -1)
			#name_weights = 'name=crossvalwithsameslidesdata;ep=15;dataaugm:hor&vert;bs=12;arch=FPN;encoder=resnet34;precision=32_val:1'    ##############
			#name_weights = 'name=TestFPN32bs;ep=21;dataaugm:hor&vert;bs=50;arch=FPN;encoder=resnet34;precision=32_val:1'    
			#name_weights = 'name=firsttestUNET;ep=30;dataaugm:hor&vert;bs=12;arch=UNET;encoder=resnet34;precision=32_val:1'
			#name_weights = 'name=FPNresnet34combinelossrestart;ep=50;bs=25;arch=FPN;encoder=resnet34;precision=32_val:1'
			#name_weights = 'name=UNETcombineloss;ep=50;bs=25;arch=UNET;encoder=resnet34;precision=32_val:1'
			#name_weights = 'name=UNETcombineloss;ep=50;bs=20;arch=UNET;encoder=resnet34;precision=32_val:2'
			#name_weights = 'name=UNETcombineloss;ep=50;bs=25;arch=UNET;encoder=resnet34;precision=32_val:3'
			name_weights = 'name=UNETcombineloss;ep=50;bs=20;arch=UNET;encoder=resnet34;precision=32_val:5'
			model.load_state_dict(torch.load(os.path.join(pipe.path, name_weights + '.pt')))    
			obj.save_pred_global(valid_folder, name_dataset, level, patch_size, overlap, pl_module = model, threshold_fill=0.1, name_exp = name_exp, mask = True, mode = 'AVG', device = 0, resized = 8) ##################resized
		
		else:
			
			#test_=notNone level=0 encoder=resnet34 arch=UNET name_exp=UNETcombinelossNegRand device=0 python test.py
			level, patch_size, arch, encoder, name_model, overlap, threshold, name_exp, device = int(os.environ.get('level')), os.environ.get('patch_size'), os.environ.get('arch'), os.environ.get('encoder'), os.environ.get('name_model'), os.environ.get('overlap'), os.environ.get('threshold'), os.environ.get('name_exp'), os.environ.get('device')
			
			patch_size = utils.os_environ(patch_size, 512, 'int')
			overlap, device = utils.os_environ(overlap, 128, 'int'), utils.os_environ(device, 0, 'int')
			arch, encoder = utils.os_environ(arch, 'UNET', 'string'), utils.os_environ(encoder, 'resnet34', 'string')
			obj = Save_Test()
			pipe = Pipeline_Testing(level, patch_size, name_dataset = '.', valid_folder = -1, gpu=0)
			model = pipe.create_model(arch = arch, encoder = encoder, save_path = pipe.path, name_model = ',', valid_folder = -1, train_dataset = '.')
			name_weights = 'name=UNETcombinelossNegRand;ep=24;bs=20;arch=UNET;encoder=resnet34;precision=32;ep:24.pt'
			model.load_state_dict(torch.load(os.path.join(pipe.path, 'test', name_weights)))    
			obj.save_pred_global(level, patch_size, overlap, pl_module = model, threshold_fill=0.1, name_exp = name_exp, mask = True, mode = 'AVG', device = device, resized = 8)
			
			
			
	else: #csv is not None, we want to compute csv to produce the FROC result 
	
		#csv_=True level=3 name_dataset='cut_[-1.0, 1.0]_numpos_86416_factor_1.0' level_pred=0 name_exp_preds=FPNcombineloss valid_folder=1 erode_iter=1 name_exp=FPNcombineloss th=0.9 python test.py
		name_exp_preds, name_dataset, th, level, name_exp, valid_folder, patch_size, level_pred, erode_iter = os.environ.get('name_exp_preds'), float(os.environ.get('th')), int(os.environ.get('level')), os.environ.get('name_exp'), int(os.environ.get('valid_folder')), os.environ.get('patch_size'), int(os.environ.get('level_pred')), int(os.environ.get('erode_iter')), os.environ.get('name_dataset')
		if patch_size is None: patch_size = 512
		else: patch_size = int(patch_size)
		obj = build_fast_predictions()
		obj.compute_csv_all(name_exp_preds, name_dataset, th, level, name_exp, valid_folder, patch_size, level_pred, erode_iter = erode_iter)
		
		
	
	
	
	
	
	
	
