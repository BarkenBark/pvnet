#%%
import os 
import sys
import yaml
sys.path.append('.')
sys.path.append('..')

from lib.ica.utils import *
from lib.ica.run_utils import calculate_epipolar_lines, ransac_keypoint_3d_hypothesis, ransac_keypoint_3d_hypothesis_test, calculate_center_3d
from lib.utils.draw_utils import visualize_lines, visualize_hypothesis_center_3d

import pickle
import yaml
from numpy import array
import numpy as np
from numpy.linalg import inv

# TEMP (REMOVE)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MeanShift


# SETTINGS
############################################################################

dataDir = '/var/www/webdav/Data/ICA/Scene1/pvnet'
className = 'tval'
nHypotheses = 1000
ransacThreshold = 0.02
resultPath = '.'
skipRansac = False
nmsSettings  = {'similariyThreshold':1/0.01,
				'scoreThreshold': 0.02*nHypotheses,
				'neighborThreshold': 0.1*nHypotheses} # TO-DO: Calculate actual values used in NMS by looking at max(linlierCounts)

# Visualization
plot3dBorders = [-1, 1,
				 -0.6, 1,
				 0, 1]


# Implicit
paths = get_work_paths(dataDir, className)





















# Load and pre-process data
################################################
motion = parse_motion(os.path.join(paths['poseOutDir'], 'motionScaled.txt'))
nCameras = len(motion)
K = parse_inner_parameters(paths['cameraPath'])

# Load points and covariance
with open('points2D.pickle', 'rb') as file:
    points2D = pickle.load(file)
with open('covariance.pickle', 'rb') as file:
	covariance = pickle.load(file)

# Pre-process
points2D_n = normalize_list(points2D, K)

# Downsample everything
k = 1 # only keep every kth camera and its points
idx = [i*k for i in range(nCameras) if i < nCameras/k]
motion = [motion[i] for i in idx]
points2D_n = [points2D_n[i] for i in idx]
















# SCRIPT
##############################################################

# Generates the same list as points2D_n, but only with one keypoint
# thisKeypoints[iCam] is of shape [nInstances, 2]
keypointIdx = 0
thisKeypoints = [keypoints[:, keypointIdx, :] if keypoints is not None else None for iCamera, keypoints in enumerate(points2D_n)] 


# Ransac for viewing ray intersection points
if skipRansac:
	hypothesisPoints = np.load(os.path.join(resultPath, 'hypotheses.npy'))
	inlierCounts = np.load(os.path.join(resultPath, 'inlierCounts.npy'))
else:
	hypothesisPoints, inlierCounts = ransac_keypoint_3d_hypothesis(motion, thisKeypoints, ransacThreshold, nHypotheses)
	#hypothesisPoints, inlierCounts = ransac_keypoint_3d_hypothesis_test(motion, thisKeypoints, ransacThreshold, 1, np.array([0,6]), np.array([0,1]), K)
	np.save(os.path.join(resultPath, 'hypotheses.npy'), hypothesisPoints)
	np.save(os.path.join(resultPath, 'inlierCounts.npy'), inlierCounts)


	
# Plot hypotheses
def get_gt_centers(posesPath, idx):
	poseData = yaml.load(open(posesPath, 'r'))
	poses = [parse_pose(poseData, 0, iPose) for iPose in idx]
	centers = np.array([pose[:,3] for pose in poses])
	return centers
indices = [1,2,3]
#centersGT = get_gt_centers(paths['posesPath'], indices)
#visualize_hypothesis_center_3d(hypothesisPoints, inlierCounts, centers=None, borders=plot3dBorders)

# Filter doublets from hypothesisPoints
hypothesisPoints = np.unique(hypothesisPoints, axis=0)


# Clustering with Mean-Shift
# Seed with the top scoring hypotheses
# meanShiftClusterer = MeanShift(bandwidth=0.03, bin_seeding=True, min_bin_freq=0.05*nHypotheses, cluster_all=False)
idxByScores = inlierCounts.argsort()[::-1]
idxByScores = idxByScores[0:len(idxByScores)//10]
seeds = hypothesisPoints[idxByScores]
meanShiftClusterer = MeanShift(bandwidth=0.02, seeds=seeds, bin_seeding=True, min_bin_freq=0.05*nHypotheses, cluster_all=False)
meanShiftClusterer.fit(hypothesisPoints)
clusterCenters = meanShiftClusterer.cluster_centers_
print(clusterCenters)
input('hold uo')

# # Clustering with DBSCAN
# dbscan = DBSCAN(eps=0.1, min_samples=4)
# labels = dbscan.fit_predict(hypothesisPoints)
# nLabels = len(set(labels)) - (1 if -1 in labels else 0)
# input('nlabels: {}'.format(nLabels))

# Non-maximum supression
# print(inlierCounts)
centers = calculate_center_3d(hypothesisPoints, inlierCounts, nmsSettings)
# centers = hypothesisPoints[clusters[clusters!=-1]]
print('Predicted {} centers.'.format(centers.shape[0]))
if centers.shape[0] != 30:
	visualize_hypothesis_center_3d(hypothesisPoints, inlierCounts, centers=clusterCenters, borders=plot3dBorders)

# for i in range(centers.shape[0]):
# 	print(centers[i])
# 	plt.hist(np.linalg.norm(centers[i] - hypothesisPoints, axis=1), bins=200, range=(0,0.1))
# 	plt.show()

# print('centersGT: ', centersGT)
# print('centersPred: ', centers)
# print('diff: ', centersGT-centersPred)

# CentersGT:
# [[-0.164492  0.027071  0.593432]
#  [-0.399997 -0.002612  0.603424]
#  [-0.417184 -0.026004  0.652383]]























