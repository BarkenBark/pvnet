# Given a trained network for vertex and segmentation prediction of a single class, 
# produce 2D coordinates of their keypoints, along with covariance matrices for the keypoints

# Add project root to path
import os
import sys
import time
sys.path.append('.')
sys.path.append('..')

# Own modules
from lib.ica.utils import *
from lib.ica.run_utils import load_network, calculate_points #run_network, calculate_hypothesis, calculate_center_2d, calculate_instance_segmentations, vertex_to_points, load_network

# Pvnet modules
from lib.networks.model_repository import Resnet18_8s
from lib.utils.data_utils import read_rgb_np
from lib.utils.draw_utils import imagenet_to_uint8, visualize_bounding_box, visualize_mask, visualize_vertex_field, visualize_overlap_mask, visualize_points, visualize_hypothesis, visualize_hypothesis_center, visualize_instance_keypoints
#import lib.ransac_voting_gpu_layer.ransac_voting as ransac_voting
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_hypothesis, ransac_voting_center


# Other modules
import torch
from PIL import Image
import numpy as np
from torch.nn import DataParallel
import torchvision.transforms as transforms # Used for converting input rgb to tensor, and normalizing to values suitable for ImageNet-trained models
import pickle

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd

# SETTINGS
#############################

# Data
dataDir = '/var/www/webdav/Data/ICA/Scenes/Deprecated/SceneDeprecated1/pvnet'
className = 'tvalgron'
formats = {'rgbFormat':'jpg', 'rgbNLeadingZeros':0}
cameraDownSampling = 10 # Keep every nth viewpoint

# Method
ransacSettings = {'nHypotheses':1024, 'threshold':0.999}
nmsSettings = {'similariyThreshold':1/15, 'scoreThreshold': 180 , 'neighborThreshold':40} # TO-DO: Calculate actual values used in NMS by looking at max(linlierCounts)
instanceSegSettings = {'thresholdMultiplier':0.9, 'discardThresholdMultiplier':0.7}

# Implicit Settings
classNameToIdx, _ = create_class_idx_dict(os.path.join(dataDir, 'models'))
paths = get_work_paths(dataDir, className, classNameToIdx)









# RUN PIPELINE
####################################

# Find number of rgb images in rgbDir
nImages = find_nbr_of_files(paths['rgbDir'], 'jpg')

#Load the network
network = load_network(paths)

# To be calculated
points2D = [] # To be filled with np arrays of shape [nInstances, nKeypoints, 2]. NOTE: nInstances is specific to each viewpoint/image
covariance = [] # To be filled with np arrays of shape [nInstances, nKeyPoints, 2, 2]. NOTE: nInstances is specific to each viewpoint/image


startTime = time.time()
for iImage in range(599,nImages):
	print(iImage)
	thisPoints2D, thisCovariance = calculate_points(iImage+1, network, paths, formats, ransacSettings, nmsSettings, instanceSegSettings)
	if (thisPoints2D is not None) & (thisCovariance is not None):
		points2D.append(thisPoints2D.cpu().detach().numpy())
		covariance.append(thisCovariance.cpu().detach().numpy())
	else:
		points2D.append(thisPoints2D)
		covariance.append(thisCovariance)
	torch.cuda.empty_cache()

	if (iImage % 50 == 0):
		elapsedTime = time.time() - startTime
		print('Processed image {}/{} after {} seconds.'.format(iImage, nImages, startTime))



# Save the results
with open('points2D.pickle', 'wb') as file:
    pickle.dump(points2D, file)

with open('covariance.pickle', 'wb') as file:
    pickle.dump(covariance, file)