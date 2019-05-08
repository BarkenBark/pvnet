# Experiment: Show that it's faster to forwardprop a batch of b data points n/b times than to forward prop one data point n times.

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
from lib.ica.run_utils import load_network, run_network, calculate_hypothesis, calculate_center, calculate_instance_segmentations, calculate_points, load_image_batch

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

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd

print('Done\n')


# SETTINGS
#############################

# Data
dataDir = '/var/www/webdav/Data/ICA/Scene1/pvnet'
className = 'tval'

# Implicit
paths = get_work_paths(dataDir, className)


# RUN PIPELINE
####################################

# Find number of rgb images in rgbDir
nImages = find_nbr_of_files(paths['rgbDir'], 'jpg')

# Regular forward loop
network = load_network(dataDir, className)

print('Forward propping one by one')
t = time.time()
for iRgb in range(nImages):
	rgb = load_image_batch(paths['rgbDir'], rgbIdx=1, batchSize=1)
	segPred, verPred = network(rgb)
print('Elapsed time: {} seconds\n'.format(time.time() - t)) # ca 16 seconds

print('Forward propping in batches')
t = time.time()
batchSize = 16
for iRgb in range(nImages//batchSize):
	rgb = load_image_batch(paths['rgbDir'], rgbIdx=1, batchSize=batchSize)
	segPred, verPred = network(rgb)
print('Elapsed time: {} seconds'.format(time.time() - t)) # ca 9 seconds