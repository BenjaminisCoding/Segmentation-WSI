import multiresolutionimageinterface as mir
import os
#import data_utils

def generate_tif_mask(tumor_names, path_to_tumor, path_to_xml, path_out): 
	
	for tumor_name in tumor_names: 
		print(tumor_name)
		pathtif = os.path.join(path_to_tumor, tumor_name + '.tif')
		pathxml = os.path.join(path_to_xml, tumor_name + '.xml')
		output_path = os.path.join(path_out, tumor_name + '_Mask.tif')
		output_path_real = os.path.join('/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask', tumor_name + '_Mask.tif')
		
		if not(os.path.exists(output_path_real)) and not(os.path.exists(output_path)): 
			reader = mir.MultiResolutionImageReader()
			mr_image = reader.open(pathtif)
			annotation_list = mir.AnnotationList()
			xml_repository = mir.XmlRepository(annotation_list)
			xml_repository.setSource(pathxml)
			xml_repository.load()
			annotation_mask = mir.AnnotationToMask()
			camelyon17_type_mask = False
			label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
			conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
			#print(output_path)
			#output_path = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask/ex.tif'
			annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)
			
def generate_tif_mask_test(test_names, test_names_minuscule, path_to_tumor, path_to_xml, path_out): 
	for i in range(len(test_names)):
		test = test_names[i]
		test_min = test_names_minuscule[i]
		print(test)
		pathtif = os.path.join(path_to_tumor, test + '.tif')
		pathxml = os.path.join(path_to_xml, test_min + '.xml')
		if os.path.exists(pathxml):
		
			output_path = os.path.join(path_out, test + '_Mask.tif')
			if not(os.path.exists(output_path)): 
				reader = mir.MultiResolutionImageReader()
				mr_image = reader.open(pathtif)
				annotation_list = mir.AnnotationList()
				xml_repository = mir.XmlRepository(annotation_list)
				xml_repository.setSource(pathxml)
				xml_repository.load()
				annotation_mask = mir.AnnotationToMask()
				camelyon17_type_mask = False
				label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
				conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
				#print(output_path)
				#output_path = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask/ex.tif'
				annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)

def get_test_names(L_mask, minuscule = False): 
	'''
	Because I often have to do this operation so to do it once it for all 
	'''
		
	names = []
	for idx in L_mask: 	
		if minuscule:
			name = 'test_'
		else:
			name = 'Test_'
		if idx < 10: name = name + f'00{idx}' 
		elif 10 <= idx < 100: name = name + f'0{idx}'
		else: name = name + f'{idx}'
		names.append(name)
	return names 



if __name__ == "__main__":
	#path_to_tumor = '/datagrid/Medical/microscopy/CAMELYON16/training/tumor'
	path_to_test = '/datagrid/Medical/microscopy/CAMELYON16/testing'
	#path_out = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/Mask'
	path_out = '/datagrid/personal/laposben/Truth/Test/test'
	#path_to_xml = '/datagrid/Medical/microscopy/CAMELYON16/Train-Ground_Truth/XML2'
	path_to_xml = '/mnt/home.stud/laposben/Documents/DATA/lesions_16_test'
	test_names = get_test_names([k for k in range(1,131)])
	test_names_minuscule = get_test_names([k for k in range(1,131)], minuscule = True) 
	#tumor_names = get_names_tumors([k for k in range(1,112)])
	generate_tif_mask_test(test_names, test_names_minuscule, path_to_test, path_to_xml, path_out)
	
	
	
	
	
	
	
	
