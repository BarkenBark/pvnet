# Test non_maximum_supression

from nms import non_maximum_supression_np

import numpy as np
import matplotlib.pyplot as plt


# Generate data
mean1 = np.array([-10,-10])
cov1 = np.array([[1,0],[0,1]])
mean2 = np.array([1,1])
cov2 = np.array([[1,-0.8],[-0.8,1]])
nSamples1 = 100
nSamples2 = 50
x1 = np.random.multivariate_normal(mean1, cov1, size=(nSamples1, ))
scores1 = 1 / np.linalg.norm(x1 - mean1, axis=1)
x2 = np.random.multivariate_normal(mean2, cov2, size=(nSamples2, ))
scores2 = 1 / np.linalg.norm(x2 - mean2, axis=1)
x = np.vstack((x1,x2))
scores = np.concatenate((scores1,scores2))


# Plot the data
plt.scatter(x[:,0], x[:,1], c=scores.ravel(), alpha=0.5)
plt.show()


# Apply non-maximum supression
detections = x
detectionScores = scores
threshold = 0.35 # Is this reasonable?
scoreThreshold = 2 # Corresponding to a distance of 0.5 from distribution mean
def similarityFun(detection, otherDetections):
	sim = 1 / np.linalg.norm((detection - otherDetections), axis=1)
	return sim

filteredIdx = non_maximum_supression_np(detections, scores, threshold, similarityFun, scoreThreshold=scoreThreshold)

print(filteredIdx)
plt.scatter(x[:,0], x[:,1], c=scores.ravel(), alpha=0.5)
plt.scatter(x[filteredIdx,0], x[filteredIdx,1], c='red')
plt.show()










