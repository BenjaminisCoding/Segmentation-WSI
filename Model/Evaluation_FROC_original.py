# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016

@author: Babak Ehteshami Bejnordi

Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys
from tqdm import tqdm

sys.path.append('/mnt/home.stud/laposben/Documents/Scripts/data')
import data_utils as utils 
   
def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.
    
    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made
        
    Returns:
        evaluation_mask
    """
    slide = openslide.open_slide(maskDIR)
    dims = slide.level_dimensions[level]
    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
    pixelarray = np.array(slide.read_region((0,0), level, dims))
    distance = nd.distance_transform_edt(np.max(pixelarray[:,:,0]) - pixelarray[:,:,0])
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    #filled_image = nd.morphology.binary_fill_holes(binary) #####################
    filled_image = nd.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2) 
    return evaluation_mask
    
    
def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    
    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object 
        should be less than 275µm to be considered as ITC (Each pixel is 
        0.243µm*0.243µm in level 0). Therefore the major axis of the object 
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.
        
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
        
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)    
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = [] 
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR,"r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        Probs.append(float(elems[0]))
        Xcorr.append(int(elems[1]))
        Ycorr.append(int(elems[2]))
    return Probs, Xcorr, Ycorr
    
         
def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image
    
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made
         
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        
        TP_probs:   A list containing the probabilities of the True positive detections
        
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
        
        detection_summary:   A python dictionary object with keys that are the labels 
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate]. 
        Lesions that are missed by the algorithm have an empty value.
        
        FP_summary:   A python dictionary object with keys that represent the 
        false positive finding number and values that contain detection 
        details [confidence score, X-coordinate, Y-coordinate]. 
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = [] 
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}  
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []        
     
    FP_counter = 0       
    if (is_tumor):
        for i in range(0,len(Xcorr)):
            HittedLabel = evaluation_mask[int(Ycorr[i]/pow(2, level)), int(Xcorr[i]/pow(2, level))]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter+=1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]                                     
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i]) 
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]] 
            FP_counter+=1
            
    num_of_tumors = max_label - len(Isolated_Tumor_Cells)
    print(num_of_tumors, len(TP_probs[TP_probs > 0]), len(FP_probs))                       
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary
 
 
def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    
    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist] 
    
    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in tqdm(all_probs[1:]):
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())    
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/max(float(sum(FROC_data[3])),1)     #in case there is no tumor
    print(np.max(total_sensitivity), 'max sensitivity')
    return  total_FPs, total_sensitivity
   
   
def plotFROC(total_FPs, total_sensitivity, final_score):
    """Plots the FROC curve
    
    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
         
    Returns:
        -
    """    
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)  
    fig.suptitle(f'Free response receiver operating characteristic curve. FROC score: {int(final_score*1e5) / 1e5}', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')    
    plt.show()       
  
#####################
### Added element ###
#####################

def compute_final_score(total_FPs, total_sensitivity):

	points = [0.25, 0.5, 1, 2, 4, 8]
	total_FPs, total_sensitivity = sorted(total_FPs), sorted(total_sensitivity)
	final_score = 0
	curr_point = 0
	for idx in range(len(total_FPs)):
	    if total_FPs[idx] > points[curr_point]:
	        final_score += total_sensitivity[idx]
	        curr_point += 1
	        if curr_point == len(points):
	            break
	while curr_point < len(points):
		final_score += total_sensitivity[-1]
		curr_point += 1
	
	return final_score / len(points)

def main(name_exp, valid_folder, level, patch_size, name_plot, th, resized, idx=None, Test = False):
	
    print(str(level), str(patch_size), 'csv', resized, name_exp, str(valid_folder), th)
    if not(Test):
        result_folder = os.path.join('/datagrid/personal/laposben/Scripts/experiments', str(level), str(patch_size), 'csv', str(resized), name_exp, str(valid_folder), str(th))
        name_plot = name_plot + ';th:' + str(th) + ';val:' + str(valid_folder)
        mask_folder = '/datagrid/personal/laposben/Truth/Mask'
    else:
    	result_folder = os.path.join('/datagrid/personal/laposben/Scripts/experiments', str(level), str(patch_size), 'csv', str(resized), name_exp, 'test', str(th))
    	name_plot = name_plot + ';th:' + str(th)
    	mask_folder = '/datagrid/personal/laposben/Truth/Test/test'
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.csv')]
    if Test:
    	result_file_list = [case for case in result_file_list if int(case[5:8]) in [79, 69, 66, 1, 11, 46, 61, 29, 26, 71, 16, 51, 10, 48, 30, 68, 40, 8, 38, 27, 73, 4, 74, 82, 64, 33, 13, 52, 84, 2, 75, 65]] 
    	
    EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243 # pixel resolution at level 0
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    
    caseNum = 0    
    print('result_file_list', result_file_list)
    for case in result_file_list:
        print('Evaluating Performance on image:', case[0:-4])
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
        if not(Test):        
            is_tumor = case[0:5] == 'tumor'    
        else:
            is_tumor = int(case[5:8]) in [79, 69, 66, 1, 11, 46, 61, 29, 26, 71, 16, 51, 10, 48, 30, 68, 40, 8, 38, 27, 73, 4, 74, 82, 64, 33, 13, 52, 84, 2, 75, 65]
        if (is_tumor):
            maskDIR = os.path.join(mask_folder, case[0:-4]) + '_Mask.tif'
            evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            evaluation_mask = 0
            ITC_labels = []
            
        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
        caseNum += 1
    
    # Compute FROC curve 
    total_FPs, total_sensitivity = computeFROC(FROC_data)
    final_score = compute_final_score(total_FPs, total_sensitivity)
    print(final_score, 'final_score')
    # plot FROC curve
    plotFROC(total_FPs, total_sensitivity, final_score)
    save_path = os.path.join('/mnt/home.stud/laposben/Documents/Scripts/experiments', str(level), str(patch_size), 'csv', name_exp)
    if not os.path.exists(save_path):
    	os.mkdir(save_path)
    if name_plot is None:
    	name_plot = f'FROC_val:{valid_folder}_noerode'
    plt.savefig(os.path.join(save_path, name_plot + '.png'))	   
    
if __name__ == "__main__":
    
    #patch_size=512 level=0 valid_folder=1 resized=8 th=0.9 name_exp= name_plot= python Evaluation_FROC_original.py
    name_exp, valid_folder, level, patch_size, name_plot, idx, th, resized = os.environ.get('name_exp'), os.environ.get('valid_folder'), int(os.environ.get('level')), int(os.environ.get('patch_size')), os.environ.get('name_plot'), os.environ.get('idx'), os.environ.get('th'), os.environ.get('resized')
    name_plot = name_plot + ';th:' + th + ';val:' + valid_folder
    mask_folder = '/datagrid/personal/laposben/Truth/Mask'
    print(str(level), str(patch_size), 'csv', resized, name_exp, str(valid_folder), th)
    result_folder = os.path.join('/datagrid/personal/laposben/Scripts/experiments', str(level), str(patch_size), 'csv', resized, name_exp, str(valid_folder), th)
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.csv')]
    if idx is not None:
    	result_file_list = [case for case in result_file_list if int(idx) == utils.str_to_idx(case[0:-4])] 
    
    EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243 # pixel resolution at level 0
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    
    caseNum = 0    
    print('result_file_list', result_file_list)
    for case in result_file_list:
        print('Evaluating Performance on image:', case[0:-4])
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
                
        is_tumor = case[0:5] == 'tumor'    
        if (is_tumor):
            maskDIR = os.path.join(mask_folder, case[0:-4]) + '_Mask.tif'
            evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            evaluation_mask = 0
            ITC_labels = []
            
        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
        caseNum += 1
    
    # Compute FROC curve 
    total_FPs, total_sensitivity = computeFROC(FROC_data)
    final_score = compute_final_score(total_FPs, total_sensitivity)
    print(final_score, 'final_score')
    # plot FROC curve
    plotFROC(total_FPs, total_sensitivity, final_score)
    save_path = os.path.join('/mnt/home.stud/laposben/Documents/Scripts/experiments', str(level), str(patch_size), 'csv', name_exp)
    if not os.path.exists(save_path):
    	os.mkdir(save_path)
    if name_plot is None:
    	name_plot = f'FROC_val:{valid_folder}_noerode'
    plt.savefig(os.path.join(save_path, name_plot + '.png'))
  
            
        
        
        
        
        
        
