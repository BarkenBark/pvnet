# Script to calculate a visibility measure for each instance for each viewpoint in a scene
# Writes results to visibility.yml

import sys
import os
sys.path.append('.')
sys.path.append('..')

# Import modules
import yaml
import time
import argparse
import numpy as np
from PIL import Image

# Import pvnet moduels
from lib.utils.data_utils import read_rgb_np

# Own moduls
from lib.ica.utils import parse_pose, parse_inner_parameters, get_work_paths, pextend, pflat, load_model_pointcloud, find_nbr_of_files






# SETTINGS
##########################################

# Parse poses path, and result path
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenedir', help='Scene pvnet directory (e.g. /path/to/Scene2/pvnet)')
args = parser.parse_args()
sceneDir = args.scenedir
assert(sceneDir is not None)

# Implicit paths
paths = get_work_paths(sceneDir)
nClasses = find_nbr_of_files(paths['modelDir'], format='ply')

# Load inner parameter matrix
K = parse_inner_parameters(paths['cameraPath'])

# Load point clouds
print('Loading pointclouds...')
tStart = time.time()
cadX = []
for iClass in range(nClasses):
	cadX.append(load_model_pointcloud(paths['modelDir'], iClass+1)[::20,:])
print('Fininshed loading pointclouds after {} seconds.'.format(time.time() - tStart))

# Parse the image width and height (assuming they all share the width/height of image 1)
rgbPath = os.path.join(paths['rgbDir'], '1.png')
rgb = read_rgb_np(rgbPath)
height, width, _ = rgb.shape


def FindNPtsWithinBorders(points, xmin, xmax, ymin, ymax):
	n = 0
	for i in range(points.shape[0]):
		p = points[i]
		if (p[0] < xmin) or (p[0] > xmax):
			continue
		elif (p[1] < ymin) or (p[1] > ymax):
			continue
		else:
			n += 1
	return n

def CalculateVisibility(data, iView, iInstance):
	# INPUT: poseData - The poseData (R, t, classIdx) of a single instance
	# Project CAD model onto image plane, count nbr of pixels within image boundaries
	pose = parse_pose(data, iView, iInstance)
	classIdx = data[iView][iInstance]['obj_id']
	X = cadX[classIdx-1]
	x = np.transpose(pflat(K@pose@np.transpose(pextend(X))))
	nPoints = x.shape[0]
	nPointsWithinBorders = FindNPtsWithinBorders(x, xmin=0, xmax=width, ymin=0, ymax=height)
	visibility = nPointsWithinBorders/nPoints
	return visibility


def CalculateVisibilityNew(data, iView, iInstance):
	# INPUT: poseData - The poseData (R, t, classIdx) of a single instance
	# Project CAD model onto image plane, count nbr of pixels within segmented area
	pose = parse_pose(data, iView, iInstance)
	classIdx = data[iView][iInstance]['obj_id']
	X = cadX[classIdx-1]
	x = np.transpose(pflat(K@pose@np.transpose(pextend(X))))
	#nPoints = x.shape[0]
	xPixels = x[:,0:2].round().astype(int)
	xPixels = np.unique(xPixels, axis=0)
	nPixels = xPixels.shape[0]
	segPath = os.path.join(sceneDir,'seg',str(iView+1).zfill(5)+'.png')
	seg = np.asarray(Image.open(segPath))
	
	# Filter out pixels out of bounds
	x = xPixels[:,0]
	y = xPixels[:,1]
	width = 640
	height = 360
	mask = np.logical_and(np.logical_and(x>=0, x<width), np.logical_and(y>=0, y<height))
	x = x[mask]
	y = y[mask]
	xPixelsNew = np.stack((x,y), axis=1)

	# Count proportion of projected points within segmentation area
	nPointsInSeg = np.sum(seg[xPixelsNew[:,1],xPixelsNew[:,0]]==(iInstance+1))
	visibility = float(nPointsInSeg/nPixels)
	
	return visibility



# START MAIN SCRIPT
##########################################################

print('Reading yaml...')
tStart = time.time()
with open(paths['posesPath'], 'r') as file:
	poseData = yaml.load(file, Loader=yaml.UnsafeLoader)
print('Fininshed reading yaml after {} seconds.'.format(time.time() - tStart))


print('Computing visibilities...')
tStart = time.time()
visibilityData = []
nViews = len(poseData)
for iView in range(nViews):
	poseDataView = poseData[iView]
	thisVisData = []
	for iInstance in range(len(poseDataView)):
		visibility = CalculateVisibilityNew(poseData, iView, iInstance)
		poseData[iView][iInstance]['visibility'] = visibility
		thisVisData.append({'visibility': visibility})
	visibilityData.append(thisVisData)
	sys.stdout.write('\rFinished view {}/{}'.format(iView, nViews))
	sys.stdout.flush()
sys.stdout.write('\n')
print('Fininshed computing visibilities after {} seconds'.format(time.time()-tStart))


visibilityPath = os.path.join(sceneDir, 'visibility.yml')
with open(visibilityPath, 'w') as file:
	yaml.dump(visibilityData, file)