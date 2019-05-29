# Run network with one image
# Use RANSAC voting for center point hypothesis
# Use NMS to find center points
# Use center points to create instance segmentations

print('Importing modules...')

# Add project root to path
import os
import sys
sys.path.append('.')
sys.path.append('..')

# Own modules
from lib.ica.utils import *
from lib.ica.run_utils import load_network, run_network, calculate_hypothesis, calculate_center_2d, calculate_instance_segmentations, vertex_to_points

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

dataDir = '/var/www/webdav/Data/ICA/Scenes/Deprecated/SceneDeprecated1/pvnet'
className = 'tvalgron'
rgbIdx = 600
formats = {'rgbFormat':'jpg', 'rgbNLeadingZeros':0}

# Implicit settings
classNameToIdx, _ = create_class_idx_dict(os.path.join(dataDir, 'models'))
paths = get_work_paths(dataDir, className, classNameToIdx)

nHypotheses = 1024
nmsSettings = {'similariyThreshold':1/15, 'scoreThreshold': 2 , 'neighborThreshold': 30 }
ransacThreshold = 0.999
instanceThresholdMultiplier = 0.9
instanceDiscardThresholdMultiplier = 0.7





# RUN PIPELINE
###############################

# Calculate network output
print('Loading network...')
network = load_network(paths)
print('Running network...')
segPred, verPred = run_network(network, paths, formats, rgbIdx=rgbIdx, batchSize=1)
print('Done\n')


# Run RANSAC for center point only
print('Running RANSAC for center keypoint...')
hypothesisPoints, inlierCounts = calculate_hypothesis(segPred, verPred, nHypotheses, ransacThreshold, keypointIdx=[0])
print('Done\n')


# Determine number of instances, and their center point coordinates
print('Running NMS...')
centers = calculate_center_2d(hypothesisPoints, inlierCounts, nmsSettings)
if centers.shape[1] == 0:
	print('huaisdhuihiuewfhueiwf')
	exit()
print('Found {} centers'.format(centers.shape[1]))
print('Done\n')

# Split segmentation into different instance masks
print('Generating instance masks...')
instanceMasks = calculate_instance_segmentations(segPred, verPred[0,0:2,:,:], centers, instanceThresholdMultiplier*ransacThreshold, instanceDiscardThresholdMultiplier)
print('Done\n')

# Run RANSAC for remaining keypoints, for each instance
print('Running RANSAC for all keypoints...')
points2D, covariance = vertex_to_points(verPred, instanceMasks, nHypotheses, ransacThreshold) # [nInstances, nKeypoints, 2], [nInstances, nKeypoints, 2, 2]
print('Done\n')

print('Points2D shape: ', points2D.shape)
print('Points 2D: ', points2D)

print('Covariance shape: ', covariance.shape)
print('Covariance: ', covariance)



# Visualizations
######################
rgbPath = paths['rgbDir'] + '/' + str(rgbIdx) + '.' + formats['rgbFormat']
img = Image.open(rgbPath)
img = np.array(img)

# Vertex field
visualize_vertex_field(verPred, segPred, keypointIdx=0)

# NMS center detections
visualize_hypothesis_center(img, hypothesisPoints[0,:,0,:].cpu().detach().numpy(), inlierCounts[0,:,0].cpu().detach().numpy(), centers)

# Overlap mask (With instances)
visualize_overlap_mask(img, instanceMasks.cpu().detach().numpy()[None,:,:,:], None)

# Plot final results
visualize_instance_keypoints(img, points2D.cpu().detach().numpy(), stretch=0.25, labelPoints=True)

