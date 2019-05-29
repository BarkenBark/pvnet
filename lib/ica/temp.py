#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:48:17 2019

@author: comvis
"""

with open('poseGlobal.pickle', 'wb') as file:
    pickle.dump(poseGlobal, file)
	
	
centers = get_centers(poseGlobal)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
centerInliers = centers[all(centers < 3,1)]

plt.close()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(centerInliers[:,0],centerInliers[:,1],centerInliers[:,2])
ax.axis('equal')