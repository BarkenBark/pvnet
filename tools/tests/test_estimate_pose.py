import os
import sys
import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from numpy import genfromtxt,array,reshape,hstack,vstack,all,any,sum,zeros, swapaxes,unique,concatenate,repeat,stack,tensordot,pad,linspace,mean,argmax
import numpy as np
import cv2
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
import itertools




sys.path.append('.')
sys.path.append('..')

from lib.ica.utils import pextend, parse_inner_parameters
from lib.ica.run_utils import estimate_pose

with open('points2D.pickle', 'rb') as file:
    points2D = pickle.load(file)
with open('covariance.pickle', 'rb') as file:
	covariance = pickle.load(file)

points3D = genfromtxt('/var/www/webdav/Data/ICA/Scenes/Train/Scene1/pvnet/poseannotation_out/keypoints/tval_keypoints.txt', delimiter=',')
points3D = np.hstack((np.zeros((3,1)),points3D))


nKeypoints = points3D.shape[1]

cameras = genfromtxt('/var/www/webdav/Data/ICA/Scenes/Train/Scene1/pvnet/poseannotation_out/motionScaled.txt', delimiter=' ')
cameras = [np.hstack((np.reshape(row[0:9],(3,3)),row[9:,None])) for row in cameras]
keepRate = 10
nCamerasFull = len(cameras)
viewpointIdx = [idx*keepRate for idx in range(nCamerasFull) if idx < nCamerasFull/keepRate]
cameras = [cameras[idx] for idx in viewpointIdx]
points2D = [points2D[idx] for idx in viewpointIdx]

cameraPath = '/var/www/webdav/Data/ICA/Scenes/Train/Scene1/pvnet/rgb/camera.yml'

cameraMatrix = parse_inner_parameters(cameraPath)

settings = {'inlierThreshold':7, 'cameraMatrix':cameraMatrix}

estimate_pose(points2D, points3D, cameras,settings)

