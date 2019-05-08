import os
import time

# Import own modules
from lib.ica.utils import * # Includes parse_3D_keypoints, etc. 
from lib.ica.nms import non_maximum_supression_np

# Import pvnet modules
from lib.networks.model_repository import Resnet18_8s
from lib.utils.data_utils import read_rgb_np
import torchvision.transforms as transforms # Used for converting input rgb to tensor, and normalizing to values suitable for ImageNet-trained models
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_hypothesis, ransac_voting_layer_v3, estimate_voting_distribution_with_mean

# Import other modules
from torch.nn import DataParallel
import torch
import torch.nn.functional as f

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd
from PIL import Image










# Function for loading a network
###################################################

def load_network(dataDir, className):
	networkDir = os.path.join(dataDir, 'network')
	networkPath = networkDir + '/' + className + '/' + className + 'Network.pth'
	rgbDir = os.path.join(dataDir, 'rgb')

	# Calculate nKeypoints
	poseOutDir = os.path.join(dataDir, 'poseannotation_out')
	keypointsPath = os.path.join(poseOutDir, 'keypoints', className+'_keypoints.txt')
	keypoints = parse_3D_keypoints(keypointsPath, addCenter=True)
	nKeypoints = keypoints.shape[1]

	# Initialize and load the network
	network = Resnet18_8s(ver_dim=nKeypoints*2, seg_dim=2)
	network = DataParallel(network).cuda()
	network.load_state_dict(torch.load(networkPath))
	network.eval()

	return network

# Function for loading a batch of images
###################################################
# Loads 'batchSize' rgb images, starting from 'rgbIdx', into a [batchSize, 3, height, width] tensor
# Assuming rgb images are formatted to be names 1,2,...nImages with nLeadingZeros leading zeros

def load_image_batch(rgbDir, rgbIdx=1, batchSize=1, nLeadingZeros=0, fileFormat='jpg'):
	# Define some shitty transform
	test_img_transforms=transforms.Compose([
			    transforms.ToTensor(), # if image.dtype is np.uint8, then it will be divided by 255
			    transforms.Normalize(mean=imageNetMean,
			                         std=imageNetStd)
			])

	# Load input images
	nImages = find_nbr_of_files(rgbDir, fileFormat)
	rgb = []
	for iRgb in range(rgbIdx, rgbIdx+batchSize):
		iRgb = (iRgb-1) % nImages + 1
		rgbPath = rgbDir + '/' + str(iRgb).zfill(nLeadingZeros) + '.' + fileFormat 
		thisRgb = read_rgb_np(rgbPath) # Np array
		thisRgb = test_img_transforms(Image.fromarray(np.ascontiguousarray(thisRgb, np.uint8)))
		rgb.append(thisRgb)

	rgb = torch.stack(rgb)
	return rgb










# Function for loading image(s), and running a network on image(s)
###############################################################################

def run_network(network, paths, formats, rgbIdx=1, batchSize=1):

	# Load image(s)
	rgb = load_image_batch(paths['rgbDir'], rgbIdx=rgbIdx, batchSize=batchSize, fileFormat=formats['rgbFormat'], nLeadingZeros=formats['rgbNLeadingZeros'])

	# Run the network
	segPred, verPred = network(rgb)

	return segPred, verPred









# Function for calculating the hypothesis points and their scores
##########################################################################

def calculate_hypothesis(segPred, verPred, nHypotheses, threshold, keypointIdx = None):

	verWeight = segPred.float().cpu().detach()
	verWeight = np.argmax(verWeight, axis=1)
	verWeight = verWeight[None,:,:,:]


	# Ransac Hypotheses
	_,nKeypoints_x2,h,w = verPred.shape
	nKeypoints = nKeypoints_x2//2
	verPredAlt = verPred.reshape([1,nKeypoints,2,h,w]).permute([0,3,4,1,2])

	# Drop all keypoints except center point
	if keypointIdx is not None:
		if isinstance(keypointIdx,int):
			verPredAlt = verPredAlt[:,:,:,[keypointIdx],:]
		else:
			verPredAlt = verPredAlt[:,:,:,keypointIdx,:]

	hypothesisPoints, inlierCounts = ransac_voting_hypothesis(verWeight.squeeze(0).cuda(), verPredAlt, nHypotheses, inlier_thresh=threshold)
	return hypothesisPoints, inlierCounts










# Function for deciding the number of centers and their locations, given hypotheses for them
#################################################################################################

def calculate_center(hypothesisPoints, inlierCounts, settings):
	x = hypothesisPoints.cpu().detach().numpy()[0,:,0,:]
	scores = inlierCounts.cpu().detach().numpy()[0,:,0]

	def similarityFun(detection, otherDetections):
		sim = 1 / np.linalg.norm((detection - otherDetections), axis=1)
		return sim

	similariyThreshold = settings['similariyThreshold']
	neighborThreshold = settings['neighborThreshold']
	scoreThreshold = settings['scoreThreshold']

	# Apply non-maximum supression
	filteredIdx = non_maximum_supression_np(x, scores, similariyThreshold, similarityFun, scoreThreshold=scoreThreshold, neighborThreshold=neighborThreshold)
	filteredPoints = x[filteredIdx,:]
	return filteredPoints.T










# Function for splitting a class segmentation into instance segmentations given a number of instance centers
####################################################################################################################

def calculate_instance_segmentations(segPred, centerVerPred, centers, threshold, discardThreshold):
	# TO-DO: Make sure we do this FOR THE ENTIRE BATCH
	
	mask = torch.argmax(segPred,dim = 1).byte()
	_,height, width = mask.shape
	centerVerPredMasked = centerVerPred.masked_select(mask)
	centerVerPredMasked = centerVerPredMasked.reshape(2,torch.sum(mask)).transpose(0,1)
	centerVerPredMasked = f.normalize(centerVerPredMasked, p=2, dim=1)

	# Useful for debugging
	# xPix = 200
	# yPix = 200
	# coordsBin = np.nonzero(mask[0]).cpu().detach().numpy()
	# iPixel = np.argwhere(np.all((coordsBin-np.array([xPix,yPix]))==0, axis=1))
	#print('(x,y) for pixel (200, 200) in new rep (after normalization).: ', centerVerPredMasked[iPixel,0].item(), ' ', centerVerPredMasked[iPixel,1].item())
	# print('(x,y) for pixel (200, 200): ', centerVerPred[0,xPix,yPix].item(), ' ', centerVerPred[1,xPix,yPix].item())
	# print('(x,y) for pixel (200, 200) after masking: ', centerVerPredMasked[0], centerVerPredMasked[1])

	coords = torch.nonzero(mask[0]).float()
	coords = coords[:, [1, 0]]
	nCenters = centers.shape[1]
	nPixels = coords.shape[0]
	mul = torch.zeros((nPixels,nCenters)).cuda()
	votes = torch.zeros((nPixels,nCenters)).cuda()
	verGTNorm = torch.zeros((nPixels,nCenters)).cuda()

	# For each center, find which pixels vote for it
	for iCenter in range(nCenters):
		verGT = torch.tensor(centers[:,iCenter:iCenter+1].T).cuda()-coords
		verGTNorm[:,iCenter:iCenter+1] = verGT.norm(2, 1, True)
		verGT = verGT / verGT.norm(2, 1, True)
		mul[:,iCenter] = torch.sum(centerVerPredMasked*verGT, dim=1)
		votes[:,iCenter] = (mul[:,iCenter]>threshold).byte()
		# print('nVotes for center {}'.format(iCenter),torch.sum(votes[:,iCenter])) # Useful for debugging
	
	# Find indices of pixels which vote for different amounts of instance centers (each case handled differently)
	nVotes = torch.sum(votes, dim=1)
	noVoteIdx = nVotes==0 # Pixels who did not vote for any center
	oneVoteIdx = nVotes==1 # Pixels who voted for 1 center
	multiVoteIdx = nVotes>=2 # Pixels who voted for 2 or more centers
	idxCenter = torch.empty((nPixels,nCenters)).byte().cuda()

	# Build the array specifyinf which pixels belong to which instance
	idxCenter[oneVoteIdx] = votes.byte()[oneVoteIdx,:] # Pixels which vote for one instance center is assigned to this
	tmp = votes.float()[multiVoteIdx,:]/verGTNorm.float()[multiVoteIdx,:]
	noVoteIdx = noVoteIdx & (torch.max(mul,dim=1)[0] > discardThreshold) # Remove pixels whose score is way off
	
	if tmp.nelement() != 0:
		idxCenter[multiVoteIdx] = (tmp == torch.max(tmp,dim=1)[0][:,None])
	if torch.sum(noVoteIdx.byte()).item() != 0:
		I = torch.eye(nCenters).byte().cuda()
		idxCenter[noVoteIdx,:] = I[torch.argmax(mul[noVoteIdx], dim=1)]

	# Print number of discarded pixels
	nDiscardedPixels = nPixels - torch.sum(torch.sum(idxCenter, dim=1)==1)
	print('Discarded {}/{} pixels'.format(nDiscardedPixels, nPixels))

	# Generate and return instance mask tensor
	instanceMask = torch.zeros((nCenters, height, width)).cuda()
	for iCenter in range(nCenters):
		idx = idxCenter[:,iCenter]
		instanceMask[iCenter,coords[idx,1].long(),coords[idx,0].long()] = 1

	return instanceMask










# Function for running the Ransac pipeline to obtain 2D coordinates of all keypoints, and their covariance matrices
##########################################################################################################################

def calculate_points(verPred, instanceMasks, nHypotheses, threshold):

	nKeypoints = verPred.shape[1]//2
	nBatch = verPred.shape[0]
	nInstances, height, width = instanceMasks.shape
	verPredAlt = verPred.permute(0,2,3,1).view(nBatch, height, width, nKeypoints, 2)

	# To be returned
	points2D = torch.zeros((nInstances, nKeypoints, 2))
	covariance = torch.zeros((nInstances, nKeypoints, 2, 2))

	for iInstance in range(nInstances):
		thisMask = instanceMasks[iInstance,:,:]
		thisMask = thisMask.unsqueeze(0) # To add batch dimension
		thisMean = ransac_voting_layer_v3(thisMask, verPredAlt, nHypotheses, inlier_thresh=threshold)
		thisMean, thisCovar = estimate_voting_distribution_with_mean(thisMask, verPredAlt, thisMean) # [b,vn,2], [b,vn,2,2]
		points2D[iInstance] = thisMean[0]
		covariance[iInstance, :, :, :] = thisCovar[0]

	return points2D, covariance










# Function for calculating the epipolar lines on a specific camera belonging to a specific keypoint of all instances in a number of viewpoints
########################################################################################################################################################

def calculate_epipolar_lines(motion, points, projectionIdx=0):
	# motion is an np array of shape [nCameras, 3, 4] containing the camera matrices of each camera
	# points is a list of nCameras tensors with shapes [nInstances, nKeypoints, 2] containing the 2D keypoint projections of all instances in the particular camera
	# or
	# points is a list of nCameras tensors with shapes [nInstances, 2] containing the 2D keypoint projections of all instances in the particular camera
	# projectionIdx is the index of the camera onto which epipolar lines from other cameras shall be projected

	# To be returned
	lines = [] # Should end up with length nCameras, where lines[projectionIdx] = None

	nCameras = motion.shape[0]
	P_proj = motion[projectionIdx] 
	A_proj = P_proj[0:3,0:3]
	for iCamera in range(nCameras):

		# There is no epipolar line of x from the same camera
		if iCamera == projectionIdx:
			lines.append(None)
			continue

		# Find the camera matrices, etc.
		thisPoints = points[iCamera]
		if thisPoints is None:
			lines.append(None)
			continue

		nInstances = points[iCamera].shape[0]
		thisPoints = points[iCamera]
		P_other = motion[iCamera]
		C_other = camera_center(P_other)
		eProj = pflat(P_proj @ pextend(C_other))
		eProjCrossMat = crossmat(eProj.ravel())

		# Calculate the epipolar line of x for each instance of x in this camera
		l = np.zeros((3, nInstances))
		for iInstance in range(nInstances):
			x = pextend(thisPoints[iInstance].reshape(2,1))
			l[:, iInstance] = (eProjCrossMat @ (A_proj @ x)).reshape((3,1)).ravel()
			if -l[0,iInstance]/l[1,iInstance] >= 2:
				print('Epipolar line of center keypoint, instance {}, view {} has high slope.'.format(iCamera, iInstance))

		lines.append(l)

	assert(len(lines)==nCameras)

	return lines











# Function for computing viewing rays corresponding to a keypoint for each instance of the keypoint in all viewpoints.
# Viewing rays are expressed in the same coordinate system as motion.
##################################################################################################################################

def compute_viewing_rays(motion, points):

	# To be returned
	viewingRays = [] # Should end up with length nCameras, where viewingRays[iCam] is either an [nInstances, 6] array, or None if there were no instances.

	return viewingRays

# Function for computing the closest point between two 3D lines
# Based on the formula from morroworks.palitri.com
##################################################################################

def compute_viewing_ray_intersection(ray1, ray2):
	A = ray1[:,0]
	a = ray1[:,1]
	B = ray2[:,0]
	b = ray2[:,1]
	c = B - A # Double check this one
	D = A + a*(-(a@b)*(b@c)+(a@c)*(b@b))/((a@a)*(b@b)-(a@b)*(a@b))
	E = B + b*(()*()-()*())/(()*()-()*())
	return (P1+P2)/2