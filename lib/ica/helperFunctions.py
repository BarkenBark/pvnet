import sys
sys.path.append('.')
sys.path.append('..')
from lib.networks.model_repository import Resnet18_8s
import os
import time

# Import own modules
#from lib.ica.utils import * # Includes parse_3D_keypoints, etc. 
from lib.ica.nms import non_maximum_suppression_np, non_maximum_suppression_np_sweep
from lib.utils.draw_utils import imagenet_to_uint8, visualize_bounding_box, visualize_mask, visualize_vertex_field, visualize_overlap_mask, visualize_points, visualize_hypothesis, visualize_hypothesis_center, visualize_instance_keypoints
from lib.ica.run_utils import load_network, run_network, calculate_hypotheses, calculate_center_2d, calculate_instance_segmentations, vertex_to_points
#Import pvnet modules
from lib.utils.data_utils import read_rgb_np
import torchvision.transforms as transforms # Used for converting input rgb to tensor, and normalizing to values suitable for ImageNet-trained models
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_hypothesis, ransac_voting_layer_v3, estimate_voting_distribution_with_mean

# Import other modules
from numpy import genfromtxt,array,reshape,hstack,vstack,all,any,sum,zeros, swapaxes,unique,concatenate,repeat,stack,tensordot,pad,linspace,mean,argmax, arange
from numpy.linalg import norm
from torch.nn import DataParallel
import torch
from torch import nn
import torch.nn.functional as f
from sklearn.cluster import DBSCAN, KMeans,MeanShift,KMeans,AgglomerativeClustering,AffinityPropagation
from sklearn.neighbors import kneighbors_graph
from scipy import ndimage, misc
import scipy
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import math
#from bayespy.nodes import Categorical, Dirichlet, Beta, Mixture, Bernoulli
#from bayespy.inference import VB
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from PIL import Image, ImageFile
from sklearn import mixture
from sklearn.neighbors import radius_neighbors_graph
# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd
from PIL import Image
from numpy import array,uint8
from lib.ica.globals import imageNetMean, imageNetStd
import matplotlib.pyplot as plt

def read_rgb_np(rgb_path):
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	img = Image.open(rgb_path).convert('RGB')
	img = np.array(img,np.uint8)
	return img


test_img_transforms=transforms.Compose([
				transforms.ToTensor(), # if image.dtype is np.uint8, then it will be divided by 255
				transforms.Normalize(mean=imageNetMean,
									 std=imageNetStd)
			])

def filter_hypothesis(hypothesisPoints, votingDirection, scoreSum, pointsDirection, votingScore, similiarityThresh, neighThresh, scoreThres, minClusterSize, returnCopies=False):
	# Filter hypothesis according to minimum number of neighbour, minimum score
	# and minimum size of cluster they belong to.
	
	if returnCopies:
		print('gagggggggg\ngagagagawwge\ngawegewagagaew\ngwagawegaweg\ngawegawgwgeaag')
		hypothesisPoints = hypothesisPoints.clone()
		votingDirection = votingDirection.clone()
		scoreSum = scoreSum.clone()
		pointsDirection = pointsDirection.clone()
		votingScore = votingScore.clone()
	
	# Keep hypotheses with enough score
	hasEnoughScore = scoreSum > scoreThres
	hypothesisPoints = hypothesisPoints[hasEnoughScore]
	votingDirection = votingDirection[hasEnoughScore]
	scoreSum = scoreSum[hasEnoughScore]
	pointsDirection = pointsDirection[hasEnoughScore]
	votingScore = votingScore[hasEnoughScore]
	
	# Keep removing hypotheses until all fulfill the different minimum thresholds.
	keepHypotheses = torch.tensor([0]).byte()
	while not torch.all(keepHypotheses):
		# Find hypotheses with enough neighbours
		
		pointsDistances = torch.from_numpy( squareform(pdist(hypothesisPoints)) ).cuda()
		votingDistances = torch.from_numpy( squareform(pdist(votingDirection)) ).cuda()
		affinity_matrix = torch.exp(-(pointsDistances*1/5)**2) * torch.exp(-(votingDistances*2)**2)
		
		adjMatrix = (affinity_matrix > similiarityThresh) & ~torch.eye(affinity_matrix.shape[0]).byte().cuda()
		
		
		nNeighbours = torch.sum(adjMatrix,dim=1)
		hasEnoughNeighbours = nNeighbours > neighThresh
		
		# Partition hypotheses into subclusters uding adjacancy matrix
		n_clusters, clusterLabels = connected_components(csgraph=adjMatrix, directed=False, return_labels=True)
		
		# Determine the size of the subclusters
		clusterLabels = torch.from_numpy(clusterLabels).cuda()
		uniqueClusterLabels = torch.unique(clusterLabels)
		clusterVotes = clusterLabels == uniqueClusterLabels[:,None]
		clusterSizes = torch.sum(clusterVotes,dim=1)
		
		# Find which clusters are big enough and keep the belonging hypotheses
		isBigCluster = clusterSizes > minClusterSize
		uniqueClusterLabelsKeep = uniqueClusterLabels[isBigCluster]
		hasEnoughClusterSize = torch.any(clusterLabels==uniqueClusterLabelsKeep[:,None],dim=0)
		
		# Keep hypotheses which are both belonging to a big enough cluster and have
		# enough neighbours
		keepHypotheses = hasEnoughNeighbours & hasEnoughClusterSize
		hypothesisPoints = hypothesisPoints[keepHypotheses]
		votingDirection = votingDirection[keepHypotheses]
		scoreSum = scoreSum[keepHypotheses]
		pointsDirection = pointsDirection[keepHypotheses]
		votingScore = votingScore[keepHypotheses]
		clusterLabels = clusterLabels[keepHypotheses]
	
	# Create weights for hypotheses according to 1/(cluster size they belong to)
	clusterWeight = torch.sum(clusterVotes.float() / clusterSizes[:,None].float(),dim=0)
	
	# Create weights for hypotheses according to their inlierCount/(total inlierCount of cluster they belong to)
	weight = clusterVotes.float() * scoreSum[None]
	scoreAndClusterWeight = torch.sum(weight / torch.sum(weight,dim=1)[:,None],dim=0)
	
	if returnCopies:
		return hypothesisPoints, votingDirection, scoreSum, pointsDirection, votingScore, adjMatrix, nNeighbours, clusterWeight, scoreAndClusterWeight, clusterLabels
	else:
		return adjMatrix, nNeighbours, clusterWeight, scoreAndClusterWeight, clusterLabels

def get_local_max_points(inlierCounts_np, adjMatrix_np):
	nFilteredHypotheses = inlierCounts_np.shape[0]
	
	isMax = np.max( (adjMatrix_np+np.eye(nFilteredHypotheses)) * inlierCounts_np[None],axis=1) == inlierCounts_np
	return isMax


def get_visible_points(verPredOrig, classMask, filterSize=8, normThresh=0.07, pointsDistanceThresh=5):
	nKeypoints = verPredOrig.shape[1]//2
	
	# Create conv layer which does mean operation over a (filterSize x filterSize) pixel area
	meanFilter = torch.nn.Conv2d(nKeypoints*2, nKeypoints*2, filterSize, stride=1, padding=filterSize//2, dilation=1,groups=nKeypoints*2, bias=False).cuda()
	meanFilter.state_dict()['weight'][:]=1/filterSize**2
	
	# Use mean filter on vertex field, exclude image border element to make 
	# shapes match (since an even filterSize should be used).
	filteredVerPred = meanFilter(verPredOrig)[:,:,:-1,:-1]
	_, _, height, width = filteredVerPred.shape
	
	maskedPixels, _ = matrixToIndices(classMask)
	maskedPixels = torch.index_select(maskedPixels,1,torch.tensor([1,0]).cuda())
	
	verPred = torch.reshape(verPredOrig,[nKeypoints,2,height,width]).squeeze()
	verPredPixels = verPred[:,:,maskedPixels[:,1],maskedPixels[:,0]]
	verPredPixels = verPredPixels/torch.norm(verPredPixels,dim=1)[:,None]
	verPredPixels = verPredPixels.permute(0,2,1)
	
	filteredVerPredAlt = torch.reshape(filteredVerPred,[nKeypoints,2,height,width])
	
	# Calculate norm of mean filtered vertex filed
	verNorms = torch.norm(filteredVerPredAlt,dim=1)
	verNorms_np = verNorms.cpu().detach().numpy()
	
	# Find pixels within segmented areas whose filtered norms are small
	visibilityMatrix = (verNorms < normThresh) & classMask
	visibilityMatrix_np = visibilityMatrix.cpu().detach().numpy()
	
	visiblePointsList = []
	
	for iKeypoint in range(nKeypoints):
		# Reshape visible pixels from matrix to point form
		#visibleClusterPoints=np.stack(np.where(visibilityMatrix_np[iKeypoint])).T
		visibleClusterPoints, _ = matrixToIndices(visibilityMatrix[iKeypoint])
		
		if len(visibleClusterPoints) == 0:
			visiblePointsList.append([])
			continue
		# Find connectivity between visible points
		adjMatrix = radius_neighbors_graph(visibleClusterPoints, radius=pointsDistanceThresh, include_self=False, mode='connectivity').toarray().astype(bool)
		
		# Cluster points that are connected, i.e. belong to the same GT point
		n_components, labels = connected_components(csgraph=adjMatrix, directed=False, return_labels=True)
		labels = torch.from_numpy(labels).cuda()
		# For each different cluster of visible points, find the one with the smallest norm
		# TODO: change to the one with most inlier counts?
		visiblePoints = torch.zeros((n_components,2))
		
		for iFeature in range(n_components):
			# Find points belonging to current cluster
			isLabel = labels==(iFeature)
			
			# Get filtered vertex norms for current cluster points                
			pointCluster = visibleClusterPoints[isLabel]
			#pointClusterNorms = verNorms[iKeypoint][pointCluster[:,0],pointCluster[:,1]]
			
			pointsDirection = getPointDirections(pointCluster, maskedPixels, normalized = True)
			votingFunction = lambda x,y: innerProductExponentiated(x,y, innerProductExponent = 1, frequencyMultiplierExponent = 0, threshold = 0.999)
			votingScore = getVotingScore(pointsDirection, verPredPixels[iKeypoint], votingFunction)
			scoreSum, _ = getScoreSum(votingScore, pointsDirection)
			
			biggestScoreIdx = torch.argmax(scoreSum)
			bestPoint = pointCluster[biggestScoreIdx]
			
			visiblePoints[iFeature] = torch.index_select(bestPoint,0,torch.tensor([1,0]).cuda())  - 0.5
			# Select point with smallest norm (deprecated)
			#verNormSMallIdx = np.argmin(pointClusterNorms)
			#visiblePoints[iFeature] = pointCluster[verNormSMallIdx,::-1] - 0.5
			
		# Create a list of arrays which holds the detected visible points
		visiblePointsList.append(visiblePoints)
	
	return visiblePointsList


def getPointsDirection(points, maskedPixels, normalized = False):
	
	pointsDirection = (points[:,None] - maskedPixels[None]).float()
	
	if normalized:
		pointsDirection = pointsDirection/(torch.norm(pointsDirection,dim=2)[:,:,None])
		
	return pointsDirection

def get_points_direction(points, maskedPixels, normalized = False):
	
	pointsDirection = (points[:,None] - maskedPixels[None]).float()
	
	if normalized:
		pointsDirection = pointsDirection/(torch.norm(pointsDirection,dim=2)[:,:,None])
		
	return pointsDirection


#pointsTot = hypothesisPointsTot.half()
#maskedPixels = maskedPixels.half()
def get_point_directions_tot(hypothesisPoints, maskedPixels, normalized = False):
	
	pointsDirection = (hypothesisPoints[:,None] - maskedPixels[:,:,None])
	
	if normalized:
		pointsDirection = pointsDirection/(torch.norm(pointsDirection,dim=2)[:,:,None])
		
	return pointsDirection

def getInnerProduct(pointsDirection, verPredPointPixels, threshold = None):
	
	innerProducts = torch.sum(pointsDirection/(torch.norm(pointsDirection,dim=2)[:,:,None]) * verPredPointPixels,dim=2)
	if threshold == None:
		return innerProducts
	else:
		return innerProducts > threshold


def get_inner_product(pointsDirection, verPredPointPixels, threshold = None):
	
	innerProducts = torch.sum(pointsDirection/(torch.norm(pointsDirection,dim=2)[:,:,None]) * verPredPointPixels,dim=2)
	if threshold == None:
		return innerProducts
	else:
		return innerProducts > threshold

def get_distance(pointsDirection, verPredPixels):
	distanceSquared = torch.norm(pointsDirection,dim=3)**2 - torch.sum(pointsDirection * verPredPixels,dim=3)**2
	_ = distanceSquared.clamp_(min=0)
	return distanceSquared

def getGaussian_distance_similarity(pointsDirection, verPredPixels,distance01=20):
	distanceSquared = get_distance(pointsDirection, verPredPixels)
	gaussianDistanceSimiliarity = torch.exp(-distanceSquared/distance01)
	return gaussianDistanceSimiliarity

#verPredPointPixels = verPredPixels[iKeypoint]
def inner_product_exponentiated(pointsDirection, verPredPointPixels, innerProductExponent = 1, frequencyMultiplierExponent = 0, threshold = None):
	# frequencyMultiplierExponent means cos(2**frequencyMultiplierExponent * x)
	
	innerProducts = getInnerProduct(pointsDirection, verPredPointPixels)
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
		return innerProductsFrequency
	else:
		return (innerProductsFrequency > threshold).float()

def get_voting_score(pointsDirection, verPredPointPixels, votingFunction):
	# Returns the score of each point from each masked pixel, 
	votingScore = votingFunction(pointsDirection, verPredPointPixels)
	return votingScore

def get_score_sum(votingScore, pointsDirection):
	scoreSum = torch.sum(votingScore,dim=1).float()
	votingDirection = torch.sum(votingScore.unsqueeze(-1).float() * pointsDirection/(torch.norm(pointsDirection,dim=2)[:,:,None]),dim=1)/scoreSum[:,None]
	return scoreSum, votingDirection


def get_cluster_centers(clusterCentersArr, nFoundCenters):
	finalCenters = np.zeros((nFoundCenters,2))
	
	for iCenter in range(nFoundCenters):
		clusterCentersArrDistance = clusterCentersArr[None]-clusterCentersArr[:,None]
		norms = np.linalg.norm(clusterCentersArrDistance,axis=2)
		inliers = norms < 3
		nInliers = np.sum(inliers,axis=1)
		maxIdx = np.argmax(nInliers)
		thisInliers = inliers[maxIdx]
		finalCenters[iCenter] = np.mean(clusterCentersArr[thisInliers],axis=0)
		clusterCentersArr = clusterCentersArr[~thisInliers]
		
	return finalCenters

def get_keypoint_votes(pointVerPredMasked,pixels,points,threshold,keypointIdx):
	
	pointVerPredMasked = f.normalize(pointVerPredMasked, p=2, dim=1)
	
	nPoints = points.shape[1]
	nPixels = pixels.shape[0]
	mul = torch.zeros((nPixels,nPoints)).cuda()
	votes = torch.zeros((nPixels,nPoints),dtype=torch.uint8).cuda()
	verGTNorm = torch.zeros((nPixels,nPoints)).cuda()
	
	for iPoint in range(nPoints):
		verGT = torch.tensor(points[:,iPoint:iPoint+1].T,dtype=torch.float32).cuda()-pixels
		verGTNorm[:,iPoint:iPoint+1] = verGT.norm(2, 1, True)
		verGT = verGT / verGT.norm(2, 1, True)
		mul[:,iPoint] = torch.sum(pointVerPredMasked*verGT, dim=1)
		votes[:,iPoint] = (mul[:,iPoint]>threshold).byte()
		
	return votes

def matrix_to_indices(matrix):
	if torch.is_tensor(matrix):
		indices = torch.nonzero(matrix)
		if len(indices) == 0:
			values = torch.tensor([],device=matrix.device)
		else:
			values = matrix[indices[:,0],indices[:,1]]
	else:
		indices = np.nonzero(matrix)
		indices = stack(indices,axis=1)
		if len(indices) == 0:
			values = np.array([])
		else:
			values = matrix[indices[:,0],indices[:,1]]
		
	return indices, values

def indices_to_matrix(indices, values,matrixShape=(640,360)):
	if torch.is_tensor(indices):
		matrix = torch.zeros(matrixShape,device=indices.device)
		matrix[indices[:,0],indices[:,1]] = values.float()
	else:
		matrix = np.zeros(matrixShape)
		matrix[indices[:,0],indices[:,1]] = values
		
	return matrix

def expand_mask(mask, expandSize=1):
	
	isNumpy = isinstance(mask, np.ndarray)
	
	if isNumpy:
		expandedMask = torch.from_numpy(mask).float().squeeze().cuda()
	else:
		expandedMask = mask.float().squeeze()
		
	filterSize=3
	meanFilter = torch.nn.Conv2d(1, 1, filterSize, stride=1, padding=filterSize//2, dilation=1,groups=1, bias=False).cuda()
	meanFilter.state_dict()['weight'][:]=1
	for i in range(expandSize):
		expandedMask = meanFilter(expandedMask[None,None]).squeeze()
	
	expandedMask = torch.reshape(expandedMask > 0,mask.shape).byte()
	
	if isNumpy:
		expandedMask = expandedMask.cpu().detach().numpy()
	
	return expandedMask


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
