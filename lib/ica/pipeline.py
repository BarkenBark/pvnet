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
from lib.ica.run_utils import run_network, calculate_hypothesis, calculate_center, calculate_instance_segmentations, calculate_points, load_network

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
dataDir = '/var/www/webdav/Data/ICA/Scene1/pvnet'
className = 'tval'
formats = {'rgbFormat':'jpg', 'rgbNLeadingZeros':0}

# Method
nHypotheses = 1024
nmsSettings = {'similariyThreshold':1/15, 'scoreThreshold': 180 , 'neighborThreshold': 40 } # TO-DO: Calculate actual values used in NMS by looking at max(linlierCounts)
ransacThreshold = 0.999
instanceThresholdMultiplier = 0.9
instanceDiscardThresholdMultiplier = 0.7
viewpointDownsampling = 10 # Keep every nth viewpoint for calculations

# Implicit Settings
paths = get_work_paths(dataDir, className)









# RUN PIPELINE
####################################

# Find number of rgb images in rgbDir
nImages = find_nbr_of_files(paths['rgbDir'], 'jpg')

#Load the network
network = load_network(dataDir, className)

# To be calculated
points2D = [] # To be filled with np arrays of shape [nInstances, nKeypoints, 2]. NOTE: nInstances is specific to each viewpoint/image
covariance = [] # To be filled with np arrays of shape [nInstances, nKeyPoints, 2, 2]. NOTE: nInstances is specific to each viewpoint/image

def run_pipeline(rgbIdx):
	# Calculate network output
	# print('Running network...')
	segPred, verPred = run_network(network, paths, formats, rgbIdx=rgbIdx, batchSize=1)
	# print('Done\n')

	# Run RANSAC for center point only
	# print('Running RANSAC for center keypoint...')
	hypothesisPoints, inlierCounts = calculate_hypothesis(segPred, verPred, nHypotheses, ransacThreshold, keypointIdx=[0])
	# print('Done\n')

	# Determine number of instances, and their center point coordinates
	# print('Running NMS...')
	centers = calculate_center_2d(hypothesisPoints, inlierCounts, nmsSettings)
	if centers.shape[1] == 0:
		return None, None
	# print('Done\n')

	# Split segmentation into different instance masks
	# print('Generating instance masks...')
	instanceMasks = calculate_instance_segmentations(segPred, verPred[0,0:2,:,:], centers, instanceThresholdMultiplier*ransacThreshold, instanceDiscardThresholdMultiplier)
	# print('Done\n')

	# Run RANSAC for remaining keypoints, for each instance
	# print('Running RANSAC for all keypoints...')
	points2D, covariance = calculate_points_2d(verPred, instanceMasks, nHypotheses, ransacThreshold) # [nInstances, nKeypoints, 2], [nInstances, nKeypoints, 2, 2]



	return points2D, covariance


startTime = time.time()
for iImage in range(nImages):
	print(iImage)
	thisPoints2D, thisCovariance = run_pipeline(iImage+1)
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