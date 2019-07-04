import os
import sys
import time
import pickle
sys.path.append('.')
sys.path.append('..')

from lib.ica.utils import * # Includes parse_3D_keypoints, etc. 

from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

clusterCentersHist = pickleLoad('/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10/clusterCentersHist.pickle')
fullClusterCenters = np.vstack(clusterCentersHist)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fullClusterCenters[:,0], fullClusterCenters[:,1], fullClusterCenters[:,2], marker='.', s=0.5, alpha=0.5)
#[ax.scatter(center[:,0], center[:,1], center[:,2]) for center in clusterCentersHist]
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

print(fullClusterCenters.shape)
ms = MeanShift(bandwidth=0.045, bin_seeding=True, min_bin_freq=200)
ms.fit(fullClusterCenters)
newCenters = ms.cluster_centers_
ax.scatter(newCenters[:,0], newCenters[:,1], newCenters[:,2])
plt.show()
print(newCenters.shape)
