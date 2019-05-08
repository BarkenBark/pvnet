# Test non_maximum_supression

import os
import sys
sys.path.append('.')

from nms import non_maximum_supression_np

import numpy as np
import matplotlib.pyplot as plt


# Load data
iKeypoint = 0
hypothesisPoints = np.load('hypothesisPoints.npy')
inlierCounts = np.load('inlierCounts.npy')


x = hypothesisPoints[0,:,iKeypoint,:]
scores = inlierCounts[0,:,iKeypoint]


# Plot the data
plt.scatter(x[:,0], x[:,1], s=1, c=scores.ravel(), alpha=0.5, marker=',')
plt.xlim(0,640)
plt.ylim(0,360)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Determine NMS settings
similariyThreshold = 1/10 # Corresponding to points within distance 10 pixels from top scoring
scoreThreshold = 800 # TO-DO: Perhaps just use a reasnable percentage of the total number of hypotheses
neighborThreshold = 50 # 50 seems to work better
def similarityFun(detection, otherDetections):
	sim = 1 / np.linalg.norm((detection - otherDetections), axis=1)
	return sim

# Apply non-maximum supression
filteredIdx = non_maximum_supression_np(x, scores, similariyThreshold, similarityFun, scoreThreshold=scoreThreshold, neighborThreshold=neighborThreshold)

print(filteredIdx)
plt.scatter(x[:,0], x[:,1], s=1, c=scores.ravel(), alpha=0.5, marker=',')
plt.scatter(x[filteredIdx,0], x[filteredIdx,1], c='red')
plt.xlim(0,640)
plt.ylim(0,360)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()










