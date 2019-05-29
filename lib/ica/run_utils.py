import os
import time
import sys

# Import own modules
from lib.ica.utils import * # Includes parse_3D_keypoints, etc. 
from lib.ica.nms import non_maximum_suppression_np, non_maximum_suppression_np_sweep


# Import pvnet modules
from lib.networks.model_repository import Resnet18_8s
from lib.utils.data_utils import read_rgb_np
import torchvision.transforms as transforms # Used for converting input rgb to tensor, and normalizing to values suitable for ImageNet-trained models
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_hypothesis, ransac_voting_layer_v3, estimate_voting_distribution_with_mean

# Import other modules
from numpy import genfromtxt,array,reshape,hstack,vstack,all,any,sum,zeros, swapaxes,unique,concatenate,repeat,stack,tensordot,pad,linspace,mean,argmax
from numpy.linalg import norm
from torch.nn import DataParallel
import torch
import torch.nn.functional as f
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MeanShift
import math

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd
from PIL import Image










# Function for loading a network
###################################################

def load_network(paths):

	# Calculate nKeypoints
	# poseOutDir = os.path.join(dataDir, 'poseannotation_out')
	# keypointsPath = os.path.join(poseOutDir, 'keypoints', className+'_keypoints.txt')
	keypoints = parse_3D_keypoints(paths['keypointsPath'], addCenter=True)
	nKeypoints = keypoints.shape[1]

	# Initialize and load the network
	network = Resnet18_8s(ver_dim=nKeypoints*2, seg_dim=2)
	network = DataParallel(network).cuda()
	network.load_state_dict(torch.load(paths['networkPath']))
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










# Function for deciding the number of 2D centers and their locations, given hypotheses for them
#################################################################################################

def calculate_center_2d(hypothesisPoints, inlierCounts, settings):
	x = hypothesisPoints.cpu().detach().numpy()[0,:,0,:]
	scores = inlierCounts.cpu().detach().numpy()[0,:,0]

	def similarityFun(detection, otherDetections):
		sim = 1 / np.linalg.norm((detection - otherDetections), axis=1)
		return sim

	similariyThreshold = settings['similariyThreshold']
	neighborThreshold = settings['neighborThreshold']
	scoreThreshold = settings['scoreThreshold']

	# Apply non-maximum supression
	filteredIdx = non_maximum_suppression_np(x, scores, similariyThreshold, similarityFun, scoreThreshold=scoreThreshold, neighborThreshold=neighborThreshold)
	filteredPoints = x[filteredIdx,:]
	return filteredPoints.T










# Function for splitting a class segmentation into instance segmentations given a number of instance centers
####################################################################################################################
#(segPred, centerVerPred, centers, threshold, discardThreshold) = (segPred, verPred[0,0:2,:,:], centers, instanceThresholdMultiplier*ransacThreshold, instanceDiscardThresholdMultiplier)
def calculate_instance_segmentations(segPred, centerVerPred, centers, threshold, discardThreshold,getMultivoteMask = False):
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

	# Build the array specifying which pixels belong to which instance
	idxCenter[oneVoteIdx] = votes.byte()[oneVoteIdx,:] # Pixels which vote for one instance center is assigned to this
	tmp = votes.float()[multiVoteIdx,:]/verGTNorm.float()[multiVoteIdx,:]
	
	noVoteIdx = noVoteIdx & (torch.max(mul,dim=1)[0] > discardThreshold) # Remove pixels whose score is way off
	
	if tmp.nelement() != 0:
		idxCenter[multiVoteIdx] = (tmp == torch.max(tmp,dim=1)[0][:,None])
	if torch.sum(noVoteIdx.byte()).item() != 0:
		I = torch.eye(nCenters).byte().cuda()
		idxCenter[noVoteIdx,:] = I[torch.argmax(mul[noVoteIdx], dim=1)]

	# Print number of discarded pixels
	#nDiscardedPixels = nPixels - torch.sum(torch.sum(idxCenter, dim=1)==1)
	#print('Discarded {}/{} pixels'.format(nDiscardedPixels, nPixels))

	# Generate and return instance mask tensor
	instanceMask = torch.zeros((nCenters, height, width)).cuda()
	for iCenter in range(nCenters):
		idx = idxCenter[:,iCenter]
		instanceMask[iCenter,coords[idx,1].long(),coords[idx,0].long()] = 1
	if getMultivoteMask:
		multivoteMask = torch.zeros((1,height, width)).cuda()
		multivoteMask[0,coords[multiVoteIdx,1].long(),coords[multiVoteIdx,0].long()] = 1
		return instanceMask, multivoteMask
	
	return instanceMask










# Function for running the pvnet Ransac pipeline on a vertex field to obtain 2D coordinates of all keypoints, and their covariance matrices
################################################################################################################################################

def vertex_to_points(verPred, instanceMasks, nHypotheses, threshold):

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



# Function for running the entire pipeline of estimating keypoint locations for all instances of a class in an image
# TO-DO: Define kw-arguments
############################################################################################################################

def calculate_points(rgbIdx, network, paths, formats, ransacSettings, nmsSettings, instanceSegSettings):
	# Extract settings
	nHypotheses, ransacThreshold = [setting for (key,setting) in ransacSettings.items()]
	instanceThresholdMultiplier, instanceDiscardThresholdMultiplier = [setting for (key,setting) in instanceSegSettings.items()]

	# Calculate network output
	segPred, verPred = run_network(network, paths, formats, rgbIdx=rgbIdx, batchSize=1)

	# Run RANSAC for center point only
	hypothesisPoints, inlierCounts = calculate_hypothesis(segPred, verPred, nHypotheses, ransacThreshold, keypointIdx=[0])
	
	# Determine number of instances, and their center point coordinates
	centers = calculate_center_2d(hypothesisPoints, inlierCounts, nmsSettings)
	if centers.shape[1] == 0:
		return None, None

	# Split segmentation into different instance masks
	instanceMasks = calculate_instance_segmentations(segPred, verPred[0,0:2,:,:], centers, instanceThresholdMultiplier*ransacThreshold, instanceDiscardThresholdMultiplier)

	# Run RANSAC for remaining keypoints, for each instance
	points2D, covariance = vertex_to_points(verPred, instanceMasks, nHypotheses, ransacThreshold) # [nInstances, nKeypoints, 2], [nInstances, nKeypoints, 2, 2]

	return points2D, covariance



# Function for calculating keypoints in multiple views
##############################################################

def calculate_points_multiple_views(viewpointIdx, network, paths, formats, ransacSettings, nmsSettings, instanceSegSettings):
	points2D = [] # To be filled with np arrays of shape [nInstances, nKeypoints, 2]. NOTE: nInstances is specific to each viewpoint/image
	covariance = [] # To be filled with np arrays of shape [nInstances, nKeyPoints, 2, 2]. NOTE: nInstances is specific to each viewpoint/image
	for iViewpoint in viewpointIdx: # NOTE: Make sure iViewpoint goes from 1,2,...
		thisPoints2D, thisCovariance = calculate_points(iViewpoint, network, paths, formats, ransacSettings, nmsSettings, instanceSegSettings)
		if (thisPoints2D is not None) & (thisCovariance is not None):
			points2D.append(thisPoints2D.cpu().detach().numpy())
			covariance.append(thisCovariance.cpu().detach().numpy())
		else:
			points2D.append(thisPoints2D)
			covariance.append(thisCovariance)
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
	E = B + b*((a@b)*(a@c)-(b@c)*(a@a))/((a@a)*(b@b)-(a@b)*(a@b))
	return (D+E)/2

# Function for computing the distances between a point x0 and nLines 3D lines
# Based on http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
#####################################################################################

def compute_3d_point_line_distance(x0, lines):
	# X - np array of shape (3,)
	# lines - np array of shape (nLines, 3, 2) where each line is defined by [location (3,), direction (3,)]
	x1 = lines[:,:,0]
	x2 = x1 + lines[:,:,1] 
	distances = np.linalg.norm(np.cross(x0-x1, x0-x2), axis=1)/np.linalg.norm(x2-x1, axis=1)
	return distances # (nLines,)

# Function for computing vieweing rays from each camera in motion to the projection of each instance of a keypoint
# The rays are expressed as 3d lines in a global coordinate system
#########################################################################################################################

def compute_viewing_rays(motion, points):
	viewingRays = np.zeros((0,3,2))
	for iCam, P in enumerate(motion):
		thisPoints = points[iCam] # Instances of the keypoints in this camera, shape (nInstances, 2)
		if thisPoints is not None:
			nInstances = thisPoints.shape[0]
			for iInstance in range(nInstances):
				x = thisPoints[iInstance]
				ray = compute_viewing_ray(P, x)
				viewingRays = np.append(viewingRays, ray[None,:,:], axis=0)

	return viewingRays # np array of shape (nViewingRays, 3, 2) 














# Function for performing Ransac to yield 3D keypoint hypotheses, and inlier counts
##############################################################################################

def ransac_keypoint_3d_hypothesis(motion, points, threshold, nIterations):
	nCameras = len(motion)

	hypotheses = np.zeros((nIterations, 3)) # A hypothesis will be a viewing ray intersection point (representing the 3D location of the specified keypoint)
	inlierCounts = np.zeros((nIterations)) # A viewing ray is an inlier to the hypothesis if its distance to the point is less than threshold

	viewingRays = compute_viewing_rays(motion, points) # np array of shape [nViewingRays, 3, 2] (Note: We no longer care about indexing)

	# Ransac
	for iIteration in range(nIterations):

		sys.stdout.write('\rRansac iteration {} of {}'.format(iIteration+1, nIterations))
		sys.stdout.flush()

		# Sample cameras (viewpoints), make sure each has an instance of the keypoint
		bothHasInstance = False
		while not bothHasInstance:
			cameraIdx = np.random.choice(nCameras, size=(2), replace=False)
			bothHasInstance = (points[cameraIdx[0]] is not None) & (points[cameraIdx[1]] is not None)

		# Sample instance for each viewpoint
		instanceIdx = np.zeros(2, dtype=int)
		for i, iCam in enumerate(cameraIdx):
			nInstances = points[iCam].shape[0]
			instanceIdx[i] = np.random.choice(nInstances)

		# Obtain rays
		x1 = points[cameraIdx[0]][instanceIdx[0]]
		P1 = motion[cameraIdx[0]]
		ray1 = compute_viewing_ray(P1, x1)
		x2 = points[cameraIdx[1]][instanceIdx[1]]
		P2 = motion[cameraIdx[1]]
		ray2 = compute_viewing_ray(P2, x2)

		# Find closest point between rays
		X = compute_viewing_ray_intersection(ray1, ray2) # Global coordinate system

		# Calculate number of inliers
		distances = compute_3d_point_line_distance(X, viewingRays)
		nInliers = np.sum(distances < threshold)

		# Store results
		hypotheses[iIteration, :] = X
		inlierCounts[iIteration] = nInliers


	print('')
	return hypotheses, inlierCounts

# Similar to function above, but let's you sample specific viewpoints, and prints more for debugging
def ransac_keypoint_3d_hypothesis_test(motion, points, threshold, nIterations, cameraIdx, instanceIdx, K):
	nCameras = len(motion)

	hypotheses = np.zeros((nIterations, 3)) # A hypothesis will be a viewing ray intersection point (representing the 3D location of the specified keypoint)
	inlierCounts = np.zeros((nIterations)) # A viewing ray is an inlier to the hypothesis if its distance to the point is less than threshold

	# Ransac
	for iIteration in range(nIterations):

		# Obtain rays
		x1 = points[cameraIdx[0]][instanceIdx[0]]
		P1 = motion[cameraIdx[0]]
		ray1 = compute_viewing_ray(P1, x1)
		x2 = points[cameraIdx[1]][instanceIdx[1]]
		P2 = motion[cameraIdx[1]]
		ray2 = compute_viewing_ray(P2, x2)
		print(' ')
		print('x1: ', K@pextend(x1))
		print('x2: ', K@pextend(x2))
		print('x1 (normalized): ', x1)
		print('x2 (normalized): ', x2)
		print('P1: ', P1)
		print('P2: ', P2)
		print('ray1: ', ray1)
		print('ray2: ', ray2)

		# Find closest point between rays
		X = compute_viewing_ray_intersection(ray1, ray2)
		print(X)

		# Calculate number of inliers
		viewingRays = compute_viewing_rays(motion, points) # np array of shape [nViewingRays, 3, 2] (Note: We no longer care about indexing)
		genRays = np.stack((ray1,ray2))
		generatorDistances = compute_3d_point_line_distance(X, genRays)
		print('genratretre: ', generatorDistances)
		distances = compute_3d_point_line_distance(X, viewingRays)
		nInliers = np.sum(distances < threshold)
		print(' ')
		print(nInliers)
		print(distances)

		# Store results
		hypotheses[iIteration, :] = X
		inlierCounts[iIteration] = nInliers

	return hypotheses, inlierCounts










# Function for deciding the number of 3D centers and their locations, given hypotheses for them
#################################################################################################

def calculate_center_3d(hypothesisPoints, inlierCounts, settings):
	# hypothesisPoints in a (nHypotheses, 3) np array
	# inlierCounts is a (nHypotheses,) np array

	def similarityFun(detection, otherDetections):
		norm = np.linalg.norm((detection - otherDetections), axis=1)
		sim = 1 / norm
		return sim

	similariyThreshold = settings['similariyThreshold']
	neighborThreshold = settings['neighborThreshold']
	scoreThreshold = settings['scoreThreshold']

	# Apply non-maximum supression
	filteredIdx = non_maximum_supression_np_alt(hypothesisPoints, inlierCounts, similariyThreshold, similarityFun, scoreThreshold=scoreThreshold, neighborThreshold=neighborThreshold)
	filteredPoints = hypothesisPoints[filteredIdx,:]
	return filteredPoints




















# Help functions for estimate_pose below
#############################################################################################

def get_pose_hypotheses(cameras, points2D, points3D, cameraMatrix):
	nViews = len(cameras)
	
	# Create list containing ndarray's of all poses in each selected view
	posesList = [calculate_pose(points2D[iView], points3D, cameraMatrix) for iView in range(nViews)]
	
	# Create list containing ndarray's of all poses in each selected view transformed into first camera
	poseListFirstCam = [[transform_pose(posesList[iView][iInstance], cameras[iView], cameras[0]) for iInstance in range(len(posesList[iView]))] for iView in range(nViews)]
	
	# Create array of all poses in first camera shape=(nPoses,3,4), and remove doubles
	poseArray = array(list(itertools.chain.from_iterable(poseListFirstCam)))
	poseArray = unique(poseArray,axis=0)
	
	return poseArray

def transform_multiple_poses(poses1, camera1, cameras):
	# poses1 - list of poses relative camera 1
	nPoses = poses1.shape[0]
	poseTransform  = concatenate((poses1,repeat(array([0,0,0,1])[None,None,:],nPoses,axis=0)),axis=1)
	camerasArr = stack(cameras)    
	posesInViews = swapaxes(tensordot(camerasArr,poseTransform, axes=([2],[1])),1,2)
	return posesInViews

def get_num_inliers(poseArray, cameras, points2D, points3DHomo, cameraMatrix, inlierThresh):
	nViews = len(cameras)
	nKeypoints = points3DHomo.shape[1]

	# Calculate the maximum number of detection instances in a single image
	maxNbrInstances = max([points2D[iView].shape[0] for iView in range(nViews)])
	
	# Transform each pose to all selected cameras
	posesInViews = transform_multiple_poses(poseArray, cameras[0], cameras)
	
	# Calculate 3D points for all poses in all views
	pointsInViews = tensordot(posesInViews,points3DHomo,axes=([3],[0]))
	cameraPointsInViews = swapaxes(tensordot(pointsInViews,cameraMatrix,axes=([2],[1])),2,3)
	
	# Calculate 2D points for all poses in all cameras (project 3D points)
	projPointsInViews = (cameraPointsInViews[:,:,0:2,:]/cameraPointsInViews[:,:,-1:,:])
	
	# Pad the views which have a smaller number of detected instances than the
	#   maximum detected with empty projections. This is to be able to convert to an array.
	zeroArr = zeros((maxNbrInstances,2,nKeypoints))
	points2DPadded=[pad(points2D[iView],((0,maxNbrInstances-points2D[iView].shape[0]),(0,0),(0,0)),'constant',constant_values=0) \
		  for iView in range(nViews)]
	
	# Create an ndarray with shape=() containing the projected points in all selected views
	#   from all poses
	points2DArr = stack(points2DPadded)
	
	# Calculate the distances between the projected points and the NN output
	distances = norm((points2DArr[:,None] - projPointsInViews[:,:,None]), axis=3)
	
	# Calculate the inliers for each pose
	inliers = any(all(distances < inlierThresh,axis=3), axis=2)
	
	# Calculate the total number of inlier for each pose
	nInliers = sum(inliers, axis=0)
	return nInliers, maxNbrInstances

def centerSimilarity(thisDetection, detections):
	# Get center and centers
	centers = get_centers(detections)
	thisCenter = get_centers(thisDetection)
	
	# Calulate similarity as the inverse of distance
	distances = norm(centers - thisCenter,axis=1)
	similarity = 1/distances
	return similarity


# Function for finding poses of each object intance in a scene, given 2D keypoint predictions from multiple viewpoints
####################################################################################################################################

def estimate_pose(points2D, points3D, cameras, settings):
	
	# Extract settings
	inlierThreshold = settings['inlierThreshold']
	cameraMatrix = settings['cameraMatrix']

	# Pre-process:
	instancesIdx = [idx for idx, point in enumerate(points2D) if point is not None] # Remove points2D at indices where None
	points2D, cameras = get_selected_data(instancesIdx, points2D, cameras)
	nViews = len(points2D)
	points2D = [swapaxes(points2D[iView],1,2) for iView in range(nViews)] # For cv2.pnp compability
	
	# Compute all pose hypotheses
	poseGlobal = get_pose_hypotheses(cameras,points2D,points3D,cameraMatrix)

	# Count nbr of inliers
	points3DHomo = pextend(points3D, 'col')
	nInliers, maxNbrInstances = get_num_inliers(poseGlobal, cameras, points2D, points3DHomo, cameraMatrix, inlierThreshold)

	# Perform NMS, sweeping over various settings
	nSimSweeps = 20
	nScoreSweeps = 50
	nNeighSweeps = 50
	
	similarityThresholdList = [1/distance for distance in linspace(0.01,0.045,nSimSweeps)]
	similarityFun=centerSimilarity

	scoreThresholdList = [score for score in linspace(1,nViews,nScoreSweeps)]
	neighborThresholdList = [neigh for neigh in linspace(1,nViews/maxNbrInstances,nNeighSweeps)]
	t = time.time()
	detectedPosesIdx = non_maximum_suppression_np_sweep(poseGlobal, nInliers, similarityThresholdList, similarityFun, scoreThresholdList, neighborThresholdList)
	#detectedPosesIdx = non_maximum_suppression_np(poseGlobal, nInliers, 1/0.02, similarityFun, 1, 1)
	print('Time to run parameter sweep:  ', time.time() - t)
	
	nDetectedPosesIdx = [len(detectedPosesIdx[iSim][iScore][iNeigh]) for iSim in range(nSimSweeps) for iScore in range(nScoreSweeps) for iNeigh in range(nNeighSweeps) if len(detectedPosesIdx[iSim][iScore][iNeigh]) > 0]
	nDetectedPosesIdxSet = set(nDetectedPosesIdx)
	
	if nDetectedPosesIdxSet == set():
		return []
	
	nDetectedPoses = max(nDetectedPosesIdxSet, key=nDetectedPosesIdx.count)

	print('Number of detected poses:  ', nDetectedPoses)
	detectedPosesIdxMatchNDetected  = [detectedPosesIdx[iSim][iScore][iNeigh] for iSim in range(nSimSweeps) for iScore in range(nScoreSweeps) for iNeigh in range(nNeighSweeps) if len(detectedPosesIdx[iSim][iScore][iNeigh]) == nDetectedPoses]
	#detectedPosesIdxMatchNDetected = [detectedPosesIdx[x][y][z] for x in range(nSweeps) for y in range(nSweeps) for z in range(nSweeps) if len(detectedPosesIdx[x][y][z]) == nDetectedPoses ]
	uniqueDetectedPoses = unique(detectedPosesIdxMatchNDetected, axis=0,return_counts=True)
	detectedPosesIdxFinal = uniqueDetectedPoses[0][argmax(uniqueDetectedPoses[1])]
# =============================================================================
# 	maxDetections = -1
# 	for nDetections in unique(nDetectedPosesIdx):
# 		detectedPosesIdxMatchNDetected = [detectedPosesIdx[x][y][z] for x in range(nSweeps) for y in range(nSweeps) for z in range(nSweeps) if len(detectedPosesIdx[x][y][z]) == nDetections]
# 		#print(detectedPosesIdxMatchNDetected)
# 		uniqueDetectedPoses = unique(detectedPosesIdxMatchNDetected, axis=0,return_counts=True)
# 		print('nDetections :',max(uniqueDetectedPoses[1]))
# 		if max(uniqueDetectedPoses[1]) > maxDetections:
# 			maxDetections = max(uniqueDetectedPoses[1])
# 			detectedPosesIdxFinal = uniqueDetectedPoses[0][argmax(uniqueDetectedPoses[1])]
# 
# =============================================================================
	print('Final detected poses index:  ', detectedPosesIdxFinal)

	# Finally, use the best detected poses
	poses = [poseGlobal[i] for i in detectedPosesIdxFinal]

	return poses 


def estimate_pose_alt(points2D, points3D, cameras, settings):
	
	# Extract settings
	inlierThreshold = settings['inlierThreshold']
	cameraMatrix = settings['cameraMatrix']

	# Pre-process:
	instancesIdx = [idx for idx, point in enumerate(points2D) if point is not None] # Remove points2D at indices where None
	points2D, cameras = get_selected_data(instancesIdx, points2D, cameras)
	nViews = len(points2D)
	points2D = [swapaxes(points2D[iView],1,2) for iView in range(nViews)] # For cv2.pnp compability
	
	# Compute all pose hypotheses
	poseGlobal = get_pose_hypotheses(cameras,points2D,points3D,cameraMatrix)

	# Count nbr of inliers
	points3DHomo = pextend(points3D, 'col')
	nInliers, maxNbrInstances = get_num_inliers(poseGlobal, cameras, points2D, points3DHomo, cameraMatrix, inlierThreshold)

	# Obtain euler angle representations of poses (alternative representations)
	nInstances = poseGlobal.shape[0]
	poseAlt = np.zeros((nInstances, 6))
	for i in range(nInstances):
		pose = poseGlobal[i]
		R = pose[0:3,0:3]
		t = pose[:,-1]
		eul = rotationMatrixToEulerAngles(R)
		poseAlt[i,:] = np.concatenate((eul,t))

	# Filter out poses whose centers are outliers or nan
	centers = poseAlt[:,3:6]
	centerNorms = np.linalg.norm(centers, axis=1)
	nanIdx = np.where(np.isnan(centerNorms))
	centerNorms[nanIdx] = -1
	poseAlt = poseAlt[(centerNorms < 100) & (centerNorms > 0), :]


	# Perform Mean-Shift clustering, sweeping over various settings
	resBand = 50
	binFreqStart = 3
	binFreqEnd = 7
	bandwidthList = list(np.linspace(0.05,0.8,resBand))
	freqList = list(range(binFreqStart, binFreqEnd+1))

	clusterPosesAltHist = []
	nCenters = []
	j=0
	for min_bin_freq in freqList:
		i=0
		for bandwidth in bandwidthList:
			meanShiftClusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=min_bin_freq, cluster_all=False)
			wasError = False
			try:
				meanShiftClusterer.fit(poseAlt)
			except ValueError:
				print('No point for bandwidth {}'.format(bandwidth))
				wasError = True
			if not wasError:
				clusterPosesAlt = meanShiftClusterer.cluster_centers_
				clusterPosesAltHist.append(clusterPosesAlt)
				nPoses.append(clusterPosesAlt.shape[0])
			else:
				nPoses.append(0)
			i+=1
		j+=1

	bestNPoses = max(set(nPoses), key=nPoses.count) # Will be a number which decides our final guess for nInstances in the scenes
	clusterPosesAlt = [clusterPosesAltHist[i] for i in range(len(clusterPosesAltHist)) if nPoses[i]==bestNPoses]
	clusterPosesAlt = array(clusterPosesAlt)
	#KMeans()
	fig = plt.figure()
	ax = Axes3D(fig)
	print(clusterPosesAlt)





	# Return the poses
	return poses 












# Function for calculating the scalar product between each gt-prediction keypoint vertex correspondence
#################################################################################################################

#def calculate_vertex_mul():


def custom_net_score(maskGT, vertexGT, vertexWeightsGT, segPred, vertexPred,threshold=0.99):
	maskPred = torch.argmax(segPred,dim = 1).byte()
	_,height, width = maskPred.shape
	nKeypoints = vertexPred.shape[0]//2
	print('vertexPred.shape  ',vertexPred.shape)
	print('maskPred.shape  ',maskPred.shape)
	print('segPred.shape  ',segPred.shape)
	print('vertexGT.shape  ',vertexGT.shape)

	nPixels = torch.sum(maskPred)
	predVector = torch.zeros((nKeypoints,nPixels,2))
	for iKeypoint in range(nKeypoints):
		predVectorTmp = vertexPred[2*iKeypoint:2*iKeypoint+2].masked_select(maskPred	)
		#print('predVector.shape  ', predVector.shape)
		predVectorTmp = predVectorTmp.reshape(2,nPixels).transpose(0,1)
		predVectorTmp = f.normalize(predVectorTmp, p=2, dim=1)
		#print('predVector.shape  ', predVector.shape)
		predVector[iKeypoint] = predVectorTmp

	
	print('predVector.shape  ',predVector.shape)
	exit()

	

	coords = torch.nonzero(maskPred[0]).float()
	coords = coords[:, [1, 0]]
	# For each keypoint, find which pixels vote for it
	for iKeypoint in range(nKeypoints):
		verGT = torch.tensor(keypoints[:,iKeypoint:iKeypoint+1].T).cuda()-coords
		verGTNorm[:,iKeypoint:iKeypoint+1] = verGT.norm(2, 1, True)
		verGT = verGT / verGT.norm(2, 1, True)
		mul[:,iKeypoint] = torch.sum(predVector[iKeypoint]*verGT, dim=1)
		votes[:,iKeypoint] = (mul[:,iKeypoint]>threshold).byte()

