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
from lib.utils.draw_utils import visualize_hypothesis_center_3d, visualize_hypothesis_center, visualize_vertex_field

# Import other modules
from numpy import genfromtxt,array,reshape,hstack,vstack,zeros,swapaxes,unique,concatenate,repeat,stack,tensordot,pad,linspace,mean,argmax
from numpy.linalg import norm
from torch.nn import DataParallel
import torch
import torch.nn.functional as f
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.measurements import label
from scipy.linalg import sqrtm

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd
from PIL import Image, ImageFile









# Function for loading a network
###################################################

def load_network(paths):

	# Assert paths contains the necessary paths
	# TODO

	# Calculate nKeypoints
	keypoints = parse_3D_keypoints(paths['keypointsPath'], addCenter=True)
	nKeypoints = keypoints.shape[1]

	# Initialize and load the network
	network = Resnet18_8s(ver_dim=nKeypoints*2, seg_dim=2)
	network = DataParallel(network).cuda()
	checkpoint = torch.load(os.path.join(paths['networkDir'], paths['className'], 'checkpoint.pth'))
	network.load_state_dict(checkpoint['network'])
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
	segPred = segPred.detach()
	verPred = verPred.detach()

	return segPred, verPred



# Function for calculating the hypothesis points for specified keypoints, and their scores. keypointIdx=None => All keypoints.
##########################################################################

def calculate_hypotheses(segPred, verPred, nHypotheses, threshold, keypointIdx = None):

	verWeight = segPred.float().cpu().detach()
	verWeight = np.argmax(verWeight, axis=1)
	verWeight = verWeight[None,:,:,:]

	# Ransac Hypotheses
	_,nKeypoints_x2,h,w = verPred.shape
	nKeypoints = nKeypoints_x2//2
	verPredAlt = verPred.reshape([1,nKeypoints,2,h,w]).permute([0,3,4,1,2])

	# Drop all keypoints except specified point
	if keypointIdx is not None:
		if isinstance(keypointIdx,int):
			verPredAlt = verPredAlt[:,:,:,[keypointIdx],:]
		else:
			verPredAlt = verPredAlt[:,:,:,keypointIdx,:]

	hypothesisPoints, inlierCounts = ransac_voting_hypothesis(verWeight.squeeze(0).cuda(), verPredAlt, nHypotheses, inlier_thresh=threshold)
	return hypothesisPoints, inlierCounts



# Function for deciding the number of 2D centers and their locations, given hypotheses for them. (TODO: Replace with function below)
#################################################################################################

def calculate_center_2d(hypothesisPoints, inlierCounts, settings):
	x = hypothesisPoints.cpu().detach().numpy()[0,:,0,:] #(nHypotheses, 2)
	scores = inlierCounts.cpu().detach().numpy()[0,:,0] #(nHypotheses,)

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



# Function for deciding the number of 2D keypoints of each type and their locations, given hypotheses for them
#################################################################################################

def calculate_keypoints_2d(hypothesisPoints, inlierCounts, settings):
	keypointHypotheses = hypothesisPoints.cpu().detach().numpy()[0,:,:,:] #(nHypotheses, nKeypoints, 2)
	hypothesisScores = inlierCounts.cpu().detach().numpy()[0,:,:] #(nHypotheses, nKeypoints)

	def similarityFun(detection, otherDetections):
		sim = 1 / np.linalg.norm((detection - otherDetections), axis=1)
		return sim

	similariyThreshold = settings['similariyThreshold']
	neighborThreshold = settings['neighborThreshold']
	scoreThreshold = settings['scoreThreshold']

	# Apply non-maximum supression
	nKeypoints = keypointHypotheses.shape[1]
	filteredPoints = [] # Will be nKeypoints long list with various number of 2D-keypoints in each entry
	for iKeypoint in range(nKeyPoints):
		x = keypointHypotheses[:,iKeypoint,:]
		scores = hypothesisScores[:,iKeypoint]
		filteredIdx = non_maximum_suppression_np(x, scores, similariyThreshold, similarityFun, scoreThreshold=scoreThreshold, neighborThreshold=neighborThreshold)
		filteredPoints.append(x[filteredIdx,:])
	return filteredPoints



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
	instanceMask = torch.zeros((nCenters, height, width)).byte().cuda()
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

# Function for calculating the vector from pixels to 2D points
####################################################################

def get_points_direction(points, maskedPixels, normalized = False):
	pointsDirection = (points[:,:,None] - maskedPixels[None, None].float()) # (nHyp, nKeypoints, nPixels, 2)
	if normalized:
		pointsDirection = pointsDirection/(torch.norm(pointsDirection,dim=3)[:,:,:,None])
	return pointsDirection

def get_score_sum(votingScore, pointsDirection):
	scoreSum = torch.sum(votingScore,dim=2).float()
	votingDirection = torch.sum(votingScore.unsqueeze(-1).float() * pointsDirection/(torch.norm(pointsDirection,dim=3)[:,:,:,None] + 1e-6),dim=2)/(scoreSum[:,:,None]+1e-6)
	# assert(not torch.isnan(votingDirection).any())
	return scoreSum, votingDirection


# Function for assigning a score and a "average direction" to each hypothesis
# Score is the sum of the scalar products between pixel voting vectors and pixel vectors directly to hypothesis point
# Direction is the (scaled) weighted average of all pixel direction pointing to the hypothesis points, weighted by the hypothesis scores
############################################################################################################################################################

def get_hypothesis_scores(hypothesisPoints, classMask, verPred, settings):
	# Input:
	# hypothesisPoints - (nbrHypothesis, nbrKeypoints, 2) vector
	# classMask - (heigh, width)
	# 
	# Output: 
	# hypothesisScores - (nHypothesis,) vector containing direction-scores
	# hypothesisDirections - (nHypothesis, 2) vector containing average direction of votes


	# if setting['votingFunction'] == 'dist':
	# 	votingFunction = lambda x,y: getGaussianDistanceSimilarity(x,y,distance01=distanceExponentialThresh)
	# elif setting['votingFunction'] == 'inner': 
	# 	votingFunction = lambda x,y: innerProductExponentiated(x, y, innerProductExponent = 1, frequencyMultiplierExponent = 0, threshold = None)


	# Extract settings
	#voting_function = settings['voting_function']
	keypointPixelRadius = settings['keypointPixelRadius']
	voting_function = lambda x,y: get_gaussian_distance_similarity(x,y,distance01=keypointPixelRadius) * get_inner_product2(x, y)
	#voting_function2 = lambda x,y: get_gaussian_distance_similarity(x,y,distance01=keypointPixelRadius) * inner_product_exponentiated(x, y, innerProductExponent = 1, frequencyMultiplierExponent = 0, threshold = None)

	# Recast verPred
	_, nKeypoints2, height, width = verPred.shape
	nKeypoints = nKeypoints2//2
	verPredAlt = torch.reshape(verPred,[nKeypoints,2,height,width]).squeeze()


	# Calculate directions from pixels to hypothesis points
	maskedPixels, _ = matrix_to_indices(classMask)
	t = time.time()
	maskedPixels = torch.index_select(maskedPixels,1,torch.tensor([1,0]).cuda())
	print('Putting maskedPixels on cuda: {} seconds'.format(time.time()-t))


	# Whatever this does
	verPredPixels = verPredAlt[:,:,maskedPixels[:,1],maskedPixels[:,0]]
	verPredPixels = verPredPixels/(torch.norm(verPredPixels,dim=1)[:,None] + 1e-6)
	verPredPixels = verPredPixels.permute(0,2,1)


	pointsDirection = get_points_direction(hypothesisPoints, maskedPixels, normalized = False)
	hypothesisPartialScores = voting_function(pointsDirection, verPredPixels)
	hypothesisScores, hypothesisDirections = get_score_sum(hypothesisPartialScores, pointsDirection)
	# t = time.time()
	# hypothesisDirections[torch.isnan(hypothesisDirections)] = 0
	# print('isnan(): {} seconds'.format(time.time()-t))



	return hypothesisScores, hypothesisDirections, hypothesisPartialScores

# Function for calculating the othogonal distance between the verPred ray and the hypothesis
######################################################################################
def get_distance(pointsDirection, verPredPixels):
	distanceSquared = torch.norm(pointsDirection,dim=3)**2 - torch.sum(pointsDirection * verPredPixels,dim=3)**2
	_ = distanceSquared.clamp_(min=0)
	return distanceSquared


# Function for finding the voting score using the distance to the hypothesis in a gaussian function.
############################################################################
def get_gaussian_distance_similarity(pointsDirection, verPredPixels,distance01=20):
	distanceSquared = get_distance(pointsDirection, verPredPixels)
	gaussianDistanceSimiliarity = torch.exp(-distanceSquared/distance01)
	return gaussianDistanceSimiliarity


################################################################################
def get_inner_product(pointsDirection, verPredPixels, threshold = None):
	
	innerProducts = torch.sum(pointsDirection/(torch.norm(pointsDirection,dim=3)[:,:,:,None]) * verPredPixels,dim=3)
	if threshold == None:
		return innerProducts
	else:
		return innerProducts > threshold

def get_inner_product2(pointsDirection, verPredPixels):
	innerProducts = torch.sum(pointsDirection/(torch.norm(pointsDirection,dim=3)[:,:,:,None]+1e-6) * verPredPixels,dim=3)
	return innerProducts.clamp_(min=0)

#######################################################################################################
def inner_product_exponentiated(pointsDirection, verPredPixels, innerProductExponent = 1, frequencyMultiplierExponent = 0, threshold = None):
	# frequencyMultiplierExponent means cos(2**frequencyMultiplierExponent * x)
	
	t = time.time()
	innerProducts = get_inner_product(pointsDirection, verPredPixels)
	print('Taking the inner product: {} seconds'.format(time.time()-t))
	t = time.time()
	frequencyMultiplier = 2**frequencyMultiplierExponent
	innerProductThresh = torch.cos(torch.tensor([math.pi / (2*frequencyMultiplier)])).cuda()
	outliers = innerProducts < innerProductThresh
	innerProductsFrequency = innerProducts
	
	# Calculate values for cos(frequencyMultiplier * x)
	for i in range(frequencyMultiplierExponent):
		innerProductsFrequency = 2*(innerProductsFrequency**2) - 1
	
	#del innerProducts
	innerProductsFrequency.masked_fill_(outliers,0)
	innerProductsFrequency = innerProductsFrequency ** innerProductExponent
	
	if threshold == None:
		print('Doing the rest: {} seconds'.format(time.time()-t))
		return innerProductsFrequency
	else:
		print('Doing the rest: {} seconds'.format(time.time()-t))
		return (innerProductsFrequency > threshold).float()


# Calculate the euclidian distance between the points in the vector x
#####################################################################
def pairwise_squared_distances(x, y=None):
	x_norm = (x**2).sum(1).view(-1, 1)
	if y is not None:
		y_t = torch.transpose(y, 0, 1)
		y_norm = (y**2).sum(1).view(1, -1)
	else:
		y_t = torch.transpose(x, 0, 1)
		y_norm = x_norm.view(1, -1)
	
	dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
	# Ensure diagonal is zero if x=y
	# if y is None:
	#     dist = dist - torch.diag(dist.diag)
	return torch.clamp(dist, 0.0, np.inf)

#######################################################################
def read_rgb_np(rgb_path):
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	img = Image.open(rgb_path).convert('RGB')
	img = np.array(img,np.uint8)
	return img


# Function for filtering a mask by finding "islands" and removing the island which are too small
#####################################################################################################
def filter_mask(mask, threshold):
	# mask - np array
	structure = np.ones((3,3))
	islandMat, nIslands = label(mask, structure)
	islandMat = islandMat[:,:,None]
	expIslands = islandMat == np.arange(1,nIslands+1)[None, None]
	islandSizes = np.sum(expIslands, axis=(0,1))
	filteredMask = np.any(expIslands[:,:,islandSizes>threshold], axis=2).astype('uint8')
	return filteredMask

def get_downsampling_mask(mask,maxNPixels):
	
	nVisiblePixels = torch.sum(mask).item()
	downSamplingStep = math.ceil(nVisiblePixels/maxNPixels)
	downSamplingMask = torch.zeros(mask.shape).byte().cuda()
	if downSamplingStep == 1:
		downSamplingMask[:] = 1
	elif downSamplingStep == 2:
		downSamplingMask[::2,::2] = 1
		downSamplingMask[1::2,1::2] = 1
	elif downSamplingStep == 3:
		downSamplingMask[::3,::3] = 1
		downSamplingMask[1::3,1::3] = 1
		downSamplingMask[2::3,2::3] = 1
	elif downSamplingStep == 4:
		downSamplingMask[::2,::2] = 1
	elif downSamplingStep < 7:
		downSamplingMask[::2,::3] = 1
	elif downSamplingStep < 10:
		downSamplingMask[::3,::3] = 1
	elif downSamplingStep < 13:
		downSamplingMask[::3,::4] = 1
	elif downSamplingStep < 17:
		downSamplingMask[::4,::4] = 1
	else:
		downSamplingMask[::5,::5] = 1
		
	return downSamplingMask & mask


def detect_keypoints(hypothesisPoints, hypothesisScores, hypothesisDirections, hypothesisPartialScores, detectionSettings, logPath=None):

	# Extract settings
	keypointPixelRadius = detectionSettings['keypointPixelRadius']
	minBinMultiplier = detectionSettings['minBinMultiplier']
	clusterMeanMaxRadius = detectionSettings['clusterMeanMaxRadius']
	
	nPixels = hypothesisPartialScores.shape[2]
	nKeypoints = hypothesisPoints.shape[1]
	nHypotheses = hypothesisPoints.shape[0]

	# Mean shift clusterer object initialization
	ms = MeanShift(bandwidth=keypointPixelRadius, bin_seeding=True, min_bin_freq=int(nHypotheses*minBinMultiplier), n_jobs=16) # meanshift to find clusters in hypotheses

	# Reshape scores
	hypothesesPointsAndDirection = torch.cat((hypothesisPoints,keypointPixelRadius*hypothesisDirections),dim=2)

	nKeypointsList = [] # number of detections per keypoint type
	clusterCenters = []
	pixelMaxHypVotingStacked = torch.zeros((nPixels,0)).cuda()
	for iKeypoint in range(nKeypoints):

		#visualize_vertex_field(verPred, classMask[None], keypointIdx=iKeypoint)
		try:
			# Find cluster centers using both hypotheses position and voting direction coordinates
			ms.fit(hypothesesPointsAndDirection[:,iKeypoint].cpu())
		except ValueError:
			clusterCenters.append(None)
			nKeypointsList.append(0)
			print('no cluster centers found')
			continue

		# Extract found cluster centers
		thisClusterCenters = ms.cluster_centers_[:,0:2]
		clusterCenters.append(thisClusterCenters)
		nKeypointsList.append(thisClusterCenters.shape[0])

		# Calculate scores of hypotheses close to cluster centers and choose the maximum ones
		clusterCenterInlierScores = (torch.norm(hypothesisPoints[None,:,iKeypoint] - torch.from_numpy(thisClusterCenters).cuda()[:,None],dim=2)<clusterMeanMaxRadius).float()*hypothesisScores[None,:,iKeypoint]
		maxHypIdx = clusterCenterInlierScores.argmax(1)

		# Stack the partial voting scores for the maximum hypotheses
		pixelMaxHypVoting = hypothesisPartialScores[maxHypIdx,iKeypoint].permute(1,0)
		# if torch.any(torch.isnan(pixelMaxHypVoting)):
		# 	print('WARNING: Nan-value encountered in pixelMaxHypVoting')
		# pixelMaxHypVoting = torch.clamp(pixelMaxHypVoting,0,20000) # Fix: Cast nan's to 0's
		pixelMaxHypVotingStacked = torch.cat((pixelMaxHypVotingStacked,pixelMaxHypVoting),dim=1)

	if logPath is not None:
		pickleSave(clusterCenters, logPath+'_clusterCenters.pickle')
		pickleSave(nKeypoints, logPath+'_nKeypointsList.pickle')

	return pixelMaxHypVotingStacked, nKeypointsList

	# hypothesesPointsAndDirection = torch.cat((hypothesisPoints,keypointPixelRadius*hypothesisDirections),dim=2)
	# #rgb_np = read_rgb_np(paths['rgbDir'] + '/' + str(rgbIdx) + '.png')
	# # TO-DO: Append list/tensor even when no points were detected
	# nKeypointsList = [] # number of detections per keypoint type
	# pixelMaxHypVotingStacked = torch.zeros(nPixels,0).cuda()
	# _, nKeypoints2, _, _ = verPred.shape
	# nKeypoints = nKeypoints2//2 
	# for iKeypoint in range(nKeypoints):
	# 	#visualize_vertex_field(verPred, classMask[None], keypointIdx=iKeypoint)
	# 	try:
	# 		# Find cluster centers using both hypotheses position and voting direction coordinates
	# 		ms.fit(hypothesesPointsAndDirection[:,iKeypoint].cpu() )
	# 	except ValueError:
	# 		nKeypointsList.append(0)
	# 		print('no cluster centers found')
	# 		continue

	# 	# Extract found cluster centers
	# 	thisClusterCenters = ms.cluster_centers_[:,0:2]
	# 	nKeypointsList.append(thisClusterCenters.shape[0])

	# 	# Calculate scores of hypotheses close to cluster centers and choose the maximum ones
	# 	clusterCenterInlierScores = (torch.norm(hypothesisPoints[None,:,iKeypoint] - torch.from_numpy(thisClusterCenters).cuda()[:,None],dim=2)<clusterMeanMaxRadius).float()*hypothesisScores[None,:,iKeypoint]
	# 	maxHypIdx = clusterCenterInlierScores.argmax(1)

	# 	# Stack the partial voting scores for the maximum hypotheses
	# 	pixelMaxHypVoting = hypothesisPartialScores[maxHypIdx,iKeypoint].permute(1,0)
	# 	pixelMaxHypVoting = torch.clamp(pixelMaxHypVoting,0,20000)
	# 	pixelMaxHypVotingStacked = torch.cat((pixelMaxHypVotingStacked,pixelMaxHypVoting),dim=1)

	# 	#visualize_hypothesis_center(rgb_np, hypothesisPoints[:,iKeypoint].cpu().numpy(), hypothesisScores[:,iKeypoint].cpu().numpy(),thisClusterCenters.T)
	# 	#visualize_hypothesis_center(rgb_np, hypothesisPoints[:,iKeypoint].cpu().numpy(), hypothesisScores[:,iKeypoint].cpu().numpy(),hypothesisPoints[maxHypIdx,iKeypoint].cpu().numpy().T)
	# print('Finished detecting keypoints: {} seconds'.format(time.time()-t))



# Function for running the entire pipeline of estimating keypoint locations for all instances of a class in an image. NOTE: With grouping keypoints to instances. Will return nKeypoints*nInstances keypoints.
# TO-DO: Define kw-arguments
############################################################################################################################

def calculate_points(rgbIdx, network, paths, formats, ransacSettings, nmsSettings, instanceSegSettings, logDir=None):
	
	# Extract settings
	nHypotheses, ransacThreshold = [setting for (key,setting) in ransacSettings.items()]
	instanceThresholdMultiplier, instanceDiscardThresholdMultiplier = [setting for (key,setting) in instanceSegSettings.items()]

	# Calculate network output
	segPred, verPred = run_network(network, paths, formats, rgbIdx=rgbIdx, batchSize=1)
	# TO-DO: Assert that the segmented area is large enough
	# TO-DO: Convert segPred to mask directly

	# Run RANSAC for center point only
	hypothesisPoints, inlierCounts = calculate_hypotheses(segPred, verPred, nHypotheses, ransacThreshold, keypointIdx=[0])
	if logDir is not None:
		pickleSave(hypothesisPoints, os.path.join(logDir, 'hypothesisPoints', str(rgbIdx)+'.pickle'))
		pickleSave(inlierCounts, os.path.join(logDir, 'inlierCounts', str(rgbIdx)+'.pickle'))
	
	# Determine number of instances, and their center point coordinates
	centers = calculate_center_2d(hypothesisPoints, inlierCounts, nmsSettings)
	if logDir is not None:
		pickleSave(centers, os.path.join(logDir, 'centers_2d', str(rgbIdx)+'.pickle'))
	if centers.shape[1] == 0:
		if logDir is not None:
			_, _, height, width = verPred.shape
			os.makedirs(os.path.join(logDir, 'instanceMasks'), exist_ok=True)
			j = Image.new('RGB', (640,360))
			j.save(os.path.join(logDir, 'instanceMasks', str(rgbIdx).zfill(5)+'.png'))
		return None, None

	# Split segmentation into different instance masks
	instanceMasks = calculate_instance_segmentations(segPred, verPred[0,0:2,:,:], centers, instanceThresholdMultiplier*ransacThreshold, instanceDiscardThresholdMultiplier)
	if logDir is not None:
		instanceMasksSave = convertInstanceSeg(instanceMasks.cpu().detach().numpy())
		j = Image.fromarray(instanceMasksSave.astype('uint8'))
		os.makedirs(os.path.join(logDir, 'instanceMasks'), exist_ok=True)
		j.save(os.path.join(logDir, 'instanceMasks', str(rgbIdx).zfill(5)+'.png'))

	# Run RANSAC for remaining keypoints, for each instance
	points2D, covariance = vertex_to_points(verPred, instanceMasks, nHypotheses, ransacThreshold) # [nInstances, nKeypoints, 2], [nInstances, nKeypoints, 2, 2]

	return points2D, covariance

# Function for running the entire pipeline of detecting keypoint locations of a class in an image. NOTE: Without grouping of keypoints to instances. Can return an arbitrary amount of keypoint, only grouped by keypoint type.
#####################################################################################################################################

def calculate_points_2(rgbIdx, network, paths, formats, ransacSettings, nmsSettings, instanceSegSettings):
	# Extract settings
	nHypotheses, ransacThreshold = [setting for (key,setting) in ransacSettings.items()]
	instanceThresholdMultiplier, instanceDiscardThresholdMultiplier = [setting for (key,setting) in instanceSegSettings.items()]

	# Calculate network output
	segPred, verPred = run_network(network, paths, formats, rgbIdx=rgbIdx, batchSize=1)

	# Run RANSAC for all keypoints
	hypothesisPoints, inlierCounts = calculate_hypotheses(segPred, verPred, nHypotheses, ransacThreshold)
	
	# Determine number of instances, and their center point coordinates
	points2D = calculate_keypoints_2d(hypothesisPoints, inlierCounts, nmsSettings) # List of length nKeypoints where each element is a (nKeypointDetections, 2) nd-array

	return points2D

def calculate_points_3(rgbIdx, network, paths, formats, ransacSettings, detectionSettings, logDir=None):

	# Extract settings
	nHypotheses, ransacThreshold = [setting for (key,setting) in ransacSettings.items()]
	#votedEnoughThresh = 0.1 # threshold for removal of pixels if the voted for a too small fraction of the keypoints

	# TO-DO: Encapsulate in some setting dict
	filterThreshold = 500 # Minimum amount of pixels in a separate masked area to be considered below
	maxNPixels = 10000 # If a the total masked area contains more than this, donwsample it

	t = time.time()
	# Calculate network output
	segPred, verPred = run_network(network, paths, formats, rgbIdx=rgbIdx, batchSize=1)
	classMask = torch.argmax(segPred, dim=1).byte().squeeze()
	#print('Finished running network: {} seconds'.format(time.time()-t))

	t = time.time()
	# Check that the mask contains enough pixels 
	classMask = torch.from_numpy(filter_mask(classMask.cpu().numpy(), filterThreshold)).cuda()
	nPixels = torch.sum(classMask)
	if nPixels < filterThreshold:
		print('No abundant enough segmentations.')
		if logDir is not None:
			_, _, height, width = verPred.shape
			os.makedirs(os.path.join(logDir, 'instanceMasks'), exist_ok=True)
			j = Image.new('RGB', (width,height))
			j.save(os.path.join(logDir, 'instanceMasks', str(rgbIdx).zfill(5)+'.png'))
		return None, None
	elif nPixels > maxNPixels: # Downsample the mask
		classMask = get_downsampling_mask(classMask, maxNPixels)
		nPixels = torch.sum(classMask)
	#print('Finished filtering the mask: {} seconds'.format(time.time()-t))


	t = time.time()
	# Calculate the hypothesis points
	hypothesisPoints, _ = calculate_hypotheses(segPred, verPred, nHypotheses, ransacThreshold, keypointIdx=None)
	hypothesisPoints = hypothesisPoints.squeeze()
	if logDir is not None:
		pickleSave(hypothesisPoints, os.path.join(logDir, 'hypothesisPoints', str(rgbIdx)+'.pickle'))
		#pickleSave(inlierCounts, os.path.join(logDir, 'inlierCounts', str(rgbIdx)+'.pickle'))
	#print('Finished calculating hypothesis points on semantic mask: {} seconds'.format(time.time()-t))

	t = time.time()
	# Get scores and voting direction for hypotheses
	hypothesisScores, hypothesisDirections, hypothesisPartialScores = get_hypothesis_scores(hypothesisPoints, classMask, verPred, detectionSettings)
	#print('Finished calculating scores, directions: {} seconds'.format(time.time()-t))

	t = time.time()
	# Detect keypoints (TO-DO: Move to function)
	###################################################
	if logDir is not None:
		detectionLogPath = os.path.join(logDir, 'semanticDetection', str(rgbIdx))
	else:
		detectionLogPath = None
	pixelMaxHypVotingStacked, nKeypointsList = detect_keypoints(hypothesisPoints, hypothesisScores, hypothesisDirections, hypothesisPartialScores, detectionSettings, logPath=detectionLogPath)
	#print('Finished detecting keypoints: {} seconds'.format(time.time()-t))

	t = time.time()
	# Calculate the number of instances in an image TO-DO: Put in function 
	nKeypointsUnique = len(np.unique(nKeypointsList))
	nTimesKeypointsExist = len(nKeypointsList)
	expectedFraction = nTimesKeypointsExist//nKeypointsUnique
	#nInstances = np.max(np.unique(nKeypointsList)[np.sum(np.array(nKeypointsList)[:,None]==np.unique(nKeypointsList)[None],axis=0)>=expectedFraction])
	nInstances = max(set(nKeypointsList), key=nKeypointsList.count)
	#print('Finished determining nInstances: {} seconds'.format(time.time()-t))
	if nInstances < 1:
		if logDir is not None:
			_, _, height, width = verPred.shape
			os.makedirs(os.path.join(logDir, 'instanceMasks'), exist_ok=True)
			j = Image.new('RGB', (width,height))
			j.save(os.path.join(logDir, 'instanceMasks', str(rgbIdx).zfill(5)+'.png'))
		return None, None



	t = time.time()
	# Instance segmentation  (TO-DO: Move to function)
	#####################################
	# # FILTERING: Find pixels which voted enoguh for keypoints and remove others
	# votedEnough = pixelMaxHypVotingStacked.sum(1)>int(np.sum(nKeypointsList)*votedEnoughThresh)
	# pixelMaxHypVotingStackedFiltered = pixelMaxHypVotingStacked[votedEnough]

	km = KMeans(n_clusters=nInstances)
	pixelMaxHypVotingStackedNormed = pixelMaxHypVotingStacked.cpu()/(pixelMaxHypVotingStacked.cpu().norm(dim=1)[:,None] + 1e-6)
	pixelsLabels_np = km.fit_predict(pixelMaxHypVotingStackedNormed)
	pixelsLabels = torch.from_numpy(pixelsLabels_np).cuda()
	num2Onehot = torch.eye(int(nInstances)+1)

	height, width = classMask.shape
	maskedPixels, _ = matrix_to_indices(classMask)
	instanceMask = indices_to_matrix(maskedPixels, pixelsLabels+1, matrixShape=(height,width))
	instanceMasks = num2Onehot[instanceMask.long()-1].permute(2,0,1).cuda()
	instanceMasks = instanceMasks[0:-1] # Drop the "no instance"-mask
	#print('Finished instance segmentation: {} seconds'.format(time.time()-t))


	#visualize_overlap_mask(rgb_np, instanceMasks[None,[0],:,:].cpu().numpy(),None)
	#visualize_hypothesis_center(rgb_np, maskedPixels.cpu().numpy()[:], (pixelsLabels+1)*10,None)

	t = time.time()
	if logDir is not None:
		instanceMasksSave = convertInstanceSeg(instanceMasks.cpu().detach().numpy())
		j = Image.fromarray(instanceMasksSave.astype('uint8'))
		os.makedirs(os.path.join(logDir, 'instanceMasks'), exist_ok=True)
		j.save(os.path.join(logDir, 'instanceMasks', str(rgbIdx).zfill(5)+'.png'))
	#print('Saved instance mask: {} seconds'.format(time.time()-t))

	t = time.time()
	# Run RANSAC for remaining keypoints, for each instance
	points2D, covariance = vertex_to_points(verPred, instanceMasks, nHypotheses, ransacThreshold) # [nInstances, nKeypoints, 2], [nInstances, nKeypoints, 2, 2]
	#print('Finished ransacing keypoints for all instances: {} seconds'.format(time.time()-t))
	#print()

	return points2D, covariance


# Function for calculating keypoints in multiple views
##############################################################

def plot_view(iView, paths, formats):
	rgbPath = paths['rgbDir'] + '/' + str(iView).zfill(formats['rgbNLeadingZeros']) + '.' + formats['rgbFormat'] 
	img = Image.open(rgbPath)
	img.show()
	#visualize_hypothesis_center(rgb, np.zeros((1,2)), np.zeros(1))


def calculate_points_multiple_views(viewpointIdx, network, paths, formats, ransacSettings, detectionSettings, plotView=False, logDir=None, verbose=True):
	
	points2D = [] # To be filled with np arrays of shape [nInstances, nKeypoints, 2]. NOTE: nInstances is specific to each viewpoint/image
	covariance = [] # To be filled with np arrays of shape [nInstances, nKeyPoints, 2, 2]. NOTE: nInstances is specific to each viewpoint/image
	for i, iViewpoint in enumerate(viewpointIdx): # NOTE: Make sure iViewpoint goes from 1,2,...  <- Whut?
		if plotView:
			plot_view(iViewpoint, paths, formats)
		thisPoints2D, thisCovariance = calculate_points_3(iViewpoint, network, paths, formats, ransacSettings, detectionSettings, logDir=logDir)
		if (thisPoints2D is not None) & (thisCovariance is not None):
			points2D.append(thisPoints2D.cpu().detach().numpy())
			covariance.append(thisCovariance.cpu().detach().numpy())
		else:
			points2D.append(thisPoints2D)
			covariance.append(thisCovariance)
		if verbose:
			nViews = len(viewpointIdx)
			#print("Finished view {}/{}".format(i, nViews))
			sys.stdout.write("\rFinished view %i/%i" % (i, nViews))
			sys.stdout.flush()
	# Save results to file
	if logDir is not None:
		pickleSave(points2D, os.path.join(logDir, 'points2D.pickle'))
		pickleSave(covariance, os.path.join(logDir, 'covariance.pickle'))

	return points2D, covariance














































# Help functions for estimate_pose below
#############################################################################################

# Converts covariance matrix to weight values for Uncertainty-PnP
def covar_to_weight(covar):
	cov_invs = []
	for vi in range(covar.shape[0]): # For every keypoint
		if covar[vi,0,0]<1e-6 or np.sum(np.isnan(covar)[vi])>0:
			cov_invs.append(np.zeros([2,2]).astype(np.float32))
			continue
		cov_inv = np.linalg.inv(sqrtm(covar[vi]))
		cov_invs.append(cov_inv)
	cov_invs = np.asarray(cov_invs)  # pn,2,2
	weights = cov_invs.reshape([-1, 4])
	weights = weights[:, (0, 1, 3)]
	return weights

def get_pose_hypotheses(cameras, points2D, points3D, covariance, cameraMatrix):
	nViews = len(cameras)
	
	# Create list containing ndarray's of all poses in each selected view
	if covariance is not None:
		posesList = [calculate_pose_uncertainty(points2D[iView], points3D, covariance[iView], cameraMatrix) for iView in range(nViews)]
	else:
		posesList = [calculate_pose(points2D[iView], points3D, cameraMatrix) for iView in range(nViews)]

	# Create list containing ndarray's of all poses in each selected view transformed into first camera
	poseListFirstCam = [[transform_pose(posesList[iView][iInstance], cameras[iView], cameras[0]) for iInstance in range(len(posesList[iView]))] for iView in range(nViews)]
	
	# Create array of all poses in first camera shape=(nPoses,3,4), and remove doubles
	poseArray = array(list(itertools.chain.from_iterable(poseListFirstCam)))
	poseArray = unique(poseArray,axis=0)
	poseListFirstCam = [np.stack(poses, axis=0) for poses in poseListFirstCam]
	
	return poseArray, poseListFirstCam

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
	inliers = np.any(np.all(distances < inlierThresh,axis=3), axis=2)
	
	# Calculate the total number of inlier for each pose
	nInliers = np.sum(inliers, axis=0)
	return nInliers, maxNbrInstances


def get_ms_bandwidth_limits(modelDir, modelIdx, useGeoCenter=True):
	_, center, minBounds, maxBounds = load_model_pointcloud(modelDir, modelIdx) 
	if not useGeoCenter:
		center = np.zeros(3)
	diffsLower = center-minBounds
	diffsUpper = maxBounds-center
	dists = np.abs(diffsUpper+diffsLower)
	bwUpper = np.min(dists)
	bwLower = 0.25*bwUpper
	return bwLower, bwUpper



def centerSimilarity(thisDetection, detections):
	# Get center and centers
	centers = get_centers(detections)
	thisCenter = get_centers(thisDetection)
	
	# Calulate similarity as the inverse of distance
	distances = norm(centers - thisCenter,axis=1)
	similarity = 1/distances
	return similarity


def calculate_pose(points2D, points3D, camera_matrix):
	# Input: 
	# points2D - ndarray, shape=(nInstances,nKeypoints,2)
	# points3D - ndarray, shape=(nKeypoints,3)

	# Output:
	# poses - ndarray, shape=(nInstances,3,4)
	
	poses = np.zeros((len(points2D),3,4))
	for iInstance, point in enumerate(points2D):
		# Solve pnp problem
		poseParam = cv2.solvePnP(objectPoints = points3D, imagePoints = point[:,None],\
									 cameraMatrix = camera_matrix, distCoeffs = None,flags = cv2.SOLVEPNP_EPNP)
		
		# Extract pose related data and calculate pose
		R = cv2.Rodrigues(poseParam[1])[0]
		t = poseParam[2]
		pose = np.hstack((R,t))
		poses[iInstance] = pose
	return poses

def calculate_pose_uncertainty(points2D, points3D, covariance, camera_matrix):
	# Input: 
	# points2D - ndarray, shape=(nInstances,nKeypoints,2)
	# points3D - ndarray, shape=(nKeypoints,3)
	# covariance - ndarray, shape=(nInstances, nKeypoints, 2, 2)

	# Output:
	# poses - ndarray, shape=(nInstances,3,4)

	poses = np.zeros((len(points2D),3,4))
	for iInstance, points in enumerate(points2D):
		covar = covariance[iInstance]
		weights = covar_to_weight(covar)
		pose = uncertainty_pnp(points, weights, points3D, camera_matrix)
		poses[iInstance] = pose

	return poses

# Function for finding poses of each object intance in a scene, given 2D keypoint predictions from multiple viewpoints
####################################################################################################################################

def estimate_pose(points2D, points3D, cameras, settings):
	
	if all(x is None for x in points2D):
		# print('Number of detected poses: ', 0)
		# print('adasbduiasdauisdhasd')
		return []

	# Extract settings
	inlierThreshold = settings['inlierThreshold']
	cameraMatrix = settings['cameraMatrix']

	# Pre-process:
	instancesIdx = [idx for idx, point in enumerate(points2D) if point is not None] # Remove points2D at indices where None
	points2D, cameras = get_selected_data(instancesIdx, points2D, cameras)
	nViews = len(points2D)
	points2D = [swapaxes(points2D[iView],1,2) for iView in range(nViews)] # For cv2.pnp compability
	
	# Compute all pose hypotheses
	poseGlobal = get_pose_hypotheses(cameras, points2D, points3D, cameraMatrix)

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
	#print('Time to run parameter sweep:  ', time.time() - t)
	
	nDetectedPosesIdx = [len(detectedPosesIdx[iSim][iScore][iNeigh]) for iSim in range(nSimSweeps) for iScore in range(nScoreSweeps) for iNeigh in range(nNeighSweeps) if len(detectedPosesIdx[iSim][iScore][iNeigh]) > 0]
	nDetectedPosesIdxSet = set(nDetectedPosesIdx)
	
	if nDetectedPosesIdxSet == set():
		# print('Number of detected poses: ', 0)
		return []
	
	nDetectedPoses = max(nDetectedPosesIdxSet, key=nDetectedPosesIdx.count)

	# print('Number of detected poses: ', nDetectedPoses)
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
	# print('Final detected poses index:  ', detectedPosesIdxFinal)

	# Finally, use the best detected poses
	poses = [poseGlobal[i] for i in detectedPosesIdxFinal]

	return poses 


def estimate_pose_center_ms(points2D, points3D, covariance, cameras, paths, settings, plotCenters=False, logDir=None):
	
	if all(x is None for x in points2D):
		return []

	# Extract settings
	inlierThreshold = settings['inlierThreshold']
	cameraMatrix = settings['cameraMatrix']
	modelIdx = settings['classIdx']
	
	# Pre-process:
	instancesIdx = [idx for idx, point in enumerate(points2D) if point is not None] # Remove points2D at indices where None
	if covariance is not None:
		points2D, cameras, covariance = get_selected_data(instancesIdx, points2D, cameras, covariance) # Does this end up as it should?
	else:
		points2D, cameras = get_selected_data(instancesIdx, points2D, cameras)
	nViews = len(points2D)
	if logDir is not None:
		pickleSave(instancesIdx, os.path.join(logDir, 'instancesIdx.pickle'))
	#points2D = [swapaxes(points2D[iView],1,2) for iView in range(nViews)] # For cv2.pnp compability
	
	# Compute all pose hypotheses
	poseGlobal, posesList = get_pose_hypotheses(cameras,points2D,points3D,covariance,cameraMatrix)
	if logDir is not None:
		pickleSave(poseGlobal, os.path.join(logDir, 'poseGlobal.pickle'))
		pickleSave(posesList, os.path.join(logDir, 'posesList.pickle'))

	# Count nbr of inliers
	#points3DHomo = pextend(points3D, 'col')
	#nInliers, maxNbrInstances = get_num_inliers(poseGlobal, cameras, points2D, points3DHomo, cameraMatrix, inlierThreshold)

	# Filter out poses whose centers are outliers or nan
	centers = poseGlobal[:,:,3]
	centerNorms = np.linalg.norm(centers, axis=1)
	nanIdx = np.where(np.isnan(centerNorms))
	centerNorms[nanIdx] = -1
	poseGlobal = poseGlobal[(centerNorms < 100) & (centerNorms > 0), :]
	centers = poseGlobal[:,:,3]
	if logDir is not None:
		pickleSave(centers, os.path.join(logDir, 'filteredPosesCenters.pickle'))

	if plotCenters:
		visualize_hypothesis_center_3d(centers)

	# Set up Mean-Shift clustering, sweeping over various settings
	resBand = settings['bandwidthSweepResolution']
	binFreqStart = max(int(settings['binFreqMultiplier']*nViews), 3)
	binFreqEnd = binFreqStart + 5
	freqList = list(range(binFreqStart, binFreqEnd+1))

	bwLower, bwUpper = get_ms_bandwidth_limits(paths['modelDir'], modelIdx, useGeoCenter=False) #TO-DO: Replace points3D with className such that limits can be hard-coded if desireable
	bandwidthList = list(np.linspace(bwLower,bwUpper,resBand))
	if logDir is not None:
		pickleSave((bwLower, bwUpper), os.path.join(logDir, 'MeanShiftBandwidths.pickle'))

	# Run Mean-Shift clustering, sweeping over various settings
	centers = poseGlobal[:,:,3]
	clusterCentersHist = []
	#nPoses = []
	j=0
	#for min_bin_freq in freqList:
	for min_bin_freq in freqList:
		i=0
		for bandwidth in bandwidthList:
			#print('Running MeanShift for min_bin_freq={}, bandwidth={}'.format(min_bin_freq, bandwidth))
			meanShiftClusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=min_bin_freq, cluster_all=False)
			wasError = False
			try:
				meanShiftClusterer.fit(centers)
			except ValueError:
				print('No point for min_bin_freq={}, bandwidth {}'.format(min_bin_freq, bandwidth))
				wasError = True
			if not wasError:
				clusterCenters = meanShiftClusterer.cluster_centers_
				clusterCentersHist.append(clusterCenters)
				#nPoses.append(clusterCenters.shape[0])
			else:
				clusterCentersHist.append(None)
				#nPoses.append(0)
			i+=1
		j+=1
	if logDir is not None:
		pickleSave(clusterCentersHist, os.path.join(logDir, 'clusterCentersHist.pickle'))
		#pickleSave(nPoses, os.path.join(logDir, 'nPoses.pickle'))

	# If no poses were found for any run, return []
	if all(x is None for x in clusterCentersHist):
		print('No instances found.')
		return []
	clusterCentersHist = [clusterCentersHist[idx] for idx, clusterCenters in enumerate(clusterCentersHist) if clusterCenters is not None]

	# Determine the correct number of clusters, and extract centers found among that number of clusters
	nRuns = len(clusterCentersHist)
	fullClusterCenters = np.vstack(clusterCentersHist)
	msFinal = MeanShift(bandwidth=bwUpper, bin_seeding=True, min_bin_freq=nRuns//20)
	try:
		msFinal.fit(fullClusterCenters)
	except ValueError:
		print('No instances found. (When clustering cluster centers)')
		return []
	instanceCenters = msFinal.cluster_centers_
	nInstances = instanceCenters.shape[0]
	#print('Final nbr of instances: ', nInstances)
	if logDir is not None:
		pickleSave(instanceCenters, os.path.join(logDir, 'instanceCenters.pickle'))

	# Obtain a list points2DInstance for each instance containing the 2D points of the particular instance
	# Assign each pose to a cluster based on center point distance. (0,1,...nInstances-1) where None means not belonging to any instance
	bestInstanceDistance = [99999]*nInstances # Length nInstances
	initPoses = [None]*nInstances # Poses to initialize optimization with
	threshold = bwUpper
	points2DInstance = [[None]*nViews for i in range(nInstances)]
	for iView in range(nViews):
		nInstancesEst = posesList[iView].shape[0]
		for iInstanceEst in range(nInstancesEst):
			centerEst = posesList[iView][iInstanceEst,:,3]
			diffs = np.linalg.norm(instanceCenters-centerEst[None], axis=1)
			iInstance = np.argmin(diffs)
			if diffs[iInstance] < threshold:
				#print('Predicted instance {} in view {} was assigned to cluster {}'.format(iInstanceEst, iView, iInstance))
				points2DInstance[iInstance][iView] = points2D[iView][iInstanceEst]
				if diffs[iInstance] < bestInstanceDistance[iInstance]:
					initPoses[iInstance] = posesList[iView][iInstanceEst]
					bestInstanceDistance[iInstance] = diffs[iInstance]

	if logDir is not None:
		pickleSave(points2DInstance, os.path.join(logDir, 'points2DInstance.pickle'))
		pickleSave(initPoses, os.path.join(logDir, 'initPoses.pickle'))
		pickleSave(bestInstanceDistance, os.path.join(logDir, 'bestInstanceDistance.pickle'))

	# Plot the 
	if plotCenters:
		visualize_hypothesis_center_3d(clusterCenters)
		fullClusterCenters = np.vstack(clusterCentersHist)
		print(fullClusterCenters.shape)
		visualize_hypothesis_center_3d(fullClusterCenters)

	# Find the poses through reprojection minimization
	poses = []
	for iInstance in range(nInstances):
		keypointsList = points2DInstance[iInstance]
		keepIdx = [idx for idx, point in enumerate(keypointsList) if point is not None]
		keypointsList = [keypointsList[idx] for idx in keepIdx]
		cameraList = [cameras[idx] for idx in keepIdx]
		modelPointsList = [points3D for idx in keepIdx]
		improvedPose = multiviewPoseEstimation(keypointsList, modelPointsList, cameraList, cameraMatrix) #TO-DO: Testa med initiallösningar från tidigare steg
		poses.append(improvedPose)
		#print('Pose of instance {}: '.format(iInstance), improvedPose)

	# Post-processing: Filter poses that are too similar
	rmIdx = []
	diffThreshold = 0.1*get_model_diameter(paths['modelDir'], modelIdx)
	for i in range(len(poses)):
		pose1 = poses[i]
		for j in range(i+1, len(poses)):
			pose2 = poses[j]
			diff = add_metric(pose1, pose2, points3D)
			if diff < diffThreshold:
				rmIdx.append(j)
	for idx in sorted(rmIdx, reverse=True):
		del poses[idx]

	print('After post-processing:')
	[print('Pose of instance {}: {}'.format(iInstance, poses[iInstance])) for iInstance in range(len(poses))]

	# Return the poses
	return poses 


def estimate_pose_6d_ms(points2D, points3D, cameras, settings):
	
	if all(x is None for x in points2D):
		# print('Number of detected poses: ', 0)
		# print('adasbduiasdauisdhasd')
		return []

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
	nPoses = []
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

