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
import matplotlib.pyplot as plt
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

dataDir = '/var/www/webdav/Data/ICA/Scenes/Validation/Scene3/pvnet'
className = 'makaroner'

formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0}

# Implicit settings
classNameToIdx, _ = create_class_idx_dict(os.path.join(dataDir, 'models'))
paths = get_work_paths(dataDir, className, classNameToIdx)
nHypotheses = 1024
nmsSettings = {'similariyThreshold':1/20, 'scoreThreshold': 80 , 'neighborThreshold': 50 }
ransacThreshold = 0.999
instanceThresholdMultiplier = 0.9
instanceDiscardThresholdMultiplier = 0.7

print('Loading network...')
network = load_network(paths)

rgbIdx = 108
currentVisualization = 'classSeg'
continueDemo = True
while continueDemo:
	
	
	
	# Visualizations
	######################
	rgbPath = paths['rgbDir'] + '/' + str(rgbIdx) + '.' + formats['rgbFormat']
	img = Image.open(rgbPath)
	img = np.array(img)
	
	# Calculate network output
	print('Running network...')
	segPred, verPred = run_network(network, paths, formats, rgbIdx=rgbIdx, batchSize=1)
	print('Done\n')
# =============================================================================
# 	
# 	I=torch.tensor(np.eye(2))
# 	rgb, mask, ver, verWeights = trainSet.__getitem__(1)
# 	segPred = (I[mask.cpu().detach()].permute(2,0,1)[None]-0.1).cuda()
# 	img = np.array(rgb.permute(1,2,0))
# 	verPred = ver[None].cuda()
# =============================================================================
	
	# Run RANSAC for center point only
	print('Running RANSAC for center keypoint...')
	hypothesisPoints, inlierCounts = calculate_hypothesis(segPred, verPred, nHypotheses, ransacThreshold, keypointIdx=[0])
	print('Done\n')
	
	
	# Determine number of instances, and their center point coordinates
	print('Running NMS...')
	centers = calculate_center_2d(hypothesisPoints, inlierCounts, nmsSettings)
	hasCenters = centers.shape[1] > 0
		
	print('Found {} centers'.format(centers.shape[1]))
	print('Done\n')
	
	
	
	if hasCenters:
		# Split segmentation into different instance masks
		print('Generating instance masks...')
		classMask = torch.argmax(segPred,dim=1)
		instanceMasks, multivoteMask = calculate_instance_segmentations(segPred, verPred[0,0:2,:,:], centers, instanceThresholdMultiplier*ransacThreshold, instanceDiscardThresholdMultiplier,getMultivoteMask=True)
		print('Done\n')
		
		# Run RANSAC for remaining keypoints, for each instance
		print('Running RANSAC for all keypoints...')
		
		points2D, covariance = vertex_to_points(verPred, instanceMasks, nHypotheses, ransacThreshold) # [nInstances, nKeypoints, 2], [nInstances, nKeypoints, 2, 2]
		print('Done\n')

	
	if not hasCenters:
		plt.imshow(img)
		plt.show()
	elif currentVisualization == 'classSeg':
		visualize_overlap_mask(img, classMask.cpu().detach().numpy()[None,:,:,:], None)
		
	elif currentVisualization == 'ver':
		# Vertex field
		visualize_vertex_field(verPred, classMask, keypointIdx=0)	
		plt.scatter(centers[0,:], centers[1,:], c='black')
	elif currentVisualization == 'hyp':
		#visualize_hypothesis(img, hypothesisPoints[0,:,0,:].cpu().detach().numpy(), inlierCounts[0,:,0].cpu().detach().numpy(), pts_target):
		visualize_hypothesis_center(img, hypothesisPoints[0,:,0,:].cpu().detach().numpy(), inlierCounts[0,:,0].cpu().detach().numpy(), None)
		
	elif currentVisualization == 'hypDet':
		# NMS center detections
		visualize_hypothesis_center(img, hypothesisPoints[0,:,0,:].cpu().detach().numpy(), inlierCounts[0,:,0].cpu().detach().numpy(), centers)
				
	elif currentVisualization == 'instSeg':	
		# Overlap mask (With instances)
		visualize_overlap_mask(img, instanceMasks.cpu().detach().numpy()[None,:,:,:], None)
	
	elif currentVisualization == 'mulSeg':
		visualize_overlap_mask(img, multivoteMask.cpu().detach().numpy()[None,:,:,:], None)
			
	elif currentVisualization.startswith('inst') and currentVisualization.endswith('Hyp') and any([i.isdigit() for i in currentVisualization]):
		iInstance = int(''.join([i for i in currentVisualization if i.isdigit()]))
		if instanceMasks.shape[0] <= iInstance:
			print('Sorry brah, enter existing instance. Plotting all instead lol')
			hypothesisPointsSeg, inlierCountsSeg = calculate_hypothesis(segPred, verPred, nHypotheses, ransacThreshold, keypointIdx=[1])
			visualize_hypothesis_center(img, hypothesisPointsSeg[0,:,0,:].cpu().detach().numpy(), inlierCountsSeg[0,:,0].cpu().detach().numpy(), None)
		else:
			hypothesisPointInstance, inlierCountsInstance = calculate_hypothesis(segPred*instanceMasks[iInstance]-0.001, verPred, nHypotheses, ransacThreshold, keypointIdx=[1])
			#centersInstance = calculate_center_2d(hypothesisPointInstance, inlierCounts, nmsSettings)
			visualize_hypothesis_center(img, hypothesisPointInstance[0,:,0,:].cpu().detach().numpy(), inlierCountsInstance[0,:,0].cpu().detach().numpy(), None)
	elif currentVisualization == 'allKeys':	
		# Plot final results
		visualize_instance_keypoints(img, points2D.cpu().detach().numpy(), stretch=0.25, labelPoints=True)	
	else:
		print('Sorry brah, enter valid inputz')
		
		
	plt.show(block=False)
	
	plt.pause(1) # <-------
	command = input(\
	   """
	   Description: input
	   Change to image number: *image number*
	   Change to Class Segmentation: 'classSeg'
	   Change to Vertex field: 'ver'
	   Change to Hypotheses: 'hyp'
	   Change to Hypotheses + detections: 'hypDet'
	   Change to Instance Segmentations: 'instSeg'
	   Change to Multiple vote Segmentation: 'mulSeg'
	   Change to single Instance Hypotheses: 'inst'*number*'Hyp'
	   Change to all keypoints for all instances: 'allKeys'
	   Exit: 'exit'
	   """)
	
	if command == 'exit':
		continueDemo = False
	elif command.isdigit():
		rgbIdx = int(command)
	else:
		currentVisualization = command
	plt.close()