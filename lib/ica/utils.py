import os
import random
import numpy as np
import itertools
import yaml
from matplotlib import colors as mcolors

# GENERALLY USEFUL
#############################################

# Variables 
pltColorsBase = list(dict(mcolors.BASE_COLORS).keys())
pltColorsCSS4 = list(dict(mcolors.CSS4_COLORS).keys())





# Functions

# Returns colorCycler which can be iterated with using 'next(coclorCycler)''
# order can either be a string ('random') or a list
# if order is a string, the number of colors can be specified with nColors
# if order is a list, the number of colors is taken to be the length of the list
# TO-DO: Better solution for seeding. Currently, seeding would affect all randomizations for the rest of the functions in this module, right?
def pltColorCycler(colormap='base', order=None, nColors=None, seed=None): 

	if colormap == 'base':
		pltColors = pltColorsBase
	elif colormap == 'css4':
		pltColors = pltColorsCSS4
	else:
		print('pltColorCycler: Please specify a valid colormap. Using base for now.')
		pltColors = pltColorsBase

	if seed is not None:
		random.seed(seed)

	if order is not None:
		if order == 'random':
			order = list(range(len(pltColors)))
			random.shuffle(order)
			if nColors is not None:
				order = [order[i] for i in range(nColors)] # Shrink the list

		assert(isinstance(order, list))
		pltColors = [pltColors[i] for i in order]

	pltColorCycler = itertools.cycle(pltColors) # next(pltColorCycler)

	return pltColorCycler



def print_attributes(*attributes, **variables):
	if variables is not None:
		for name, variable in variables.items():
			for attribute in attributes:
				print('{}: \n{}'.format(name+'.'+attribute, getattr(variable, attribute)))
			print()



def find_nbr_of_files(directory, format=None):
	n = 0
	for item in os.listdir(directory):
		if os.path.isfile(os.path.join(directory, item)):
			if format is not None:
				if item.endswith(format):
					n += 1
	return n




















# USEFUL FOR COMPUTER VISION
###################################################################

def pflat(x):
	return x/x[-1,:]

def pextend(x):
	# Assumes each point is defined as a column
	onePad = np.ones((1,x.shape[1]))
	return np.vstack((x,onePad))

def crossmat(x):
	return np.array([[0,-x[2],x[1]], [x[2],0,-x[0]], [-x[1],x[0],0]])

def camera_center(P):
	R = P[0:3,0:3]
	t = P[:,3]
	C = (-R@t).reshape((3,1))
	return C

def normalize(x, K):
	# Assume x is [nPoints, 2] (projection)
	xn = np.linalg.inv(K)@np.transpose(np.insert(x, 2, 1, axis=1))
	xn = np.transpose(xn)[:,0:2]
	return xn

def normalize_list(points, K):
	pointsNormalized = [] # To be filled with nCameras arrays of shape [nInstances, nKeypoints, 2]
	nCameras = len(points)
	for iCam in range(nCameras):
		thisPoints = points[iCam]
		if thisPoints is None:
			print(iCam)
			print('Nonononone')
			pointsNormalized.append(None)
			continue
		nInstances = thisPoints.shape[0]
		thisPointsNormalized = np.zeros(thisPoints.shape)
		for iInstance in range(nInstances):
			x = thisPoints[iInstance]
			thisPointsNormalized[iInstance] = normalize(x, K)
		pointsNormalized.append(thisPointsNormalized)
	return pointsNormalized




# USEFUL ONLY FOR THIS PROJECT
#############################################

def get_work_paths(dataDir, className):
	paths = dict()
	paths['dataDir'] = dataDir
	paths['networkDir'] = os.path.join(dataDir, 'network')
	paths['modelDir'] = os.path.join(dataDir, 'models')
	paths['poseOutDir'] = os.path.join(dataDir, 'poseannotation_out')
	paths['rgbDir'] = os.path.join(dataDir, 'rgb')
	paths['segDir'] = os.path.join(dataDir, 'seg')
	paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
	paths['keypointsPath'] = os.path.join(paths['poseOutDir'], 'keypoints', className+'_keypoints.txt')
	paths['posesPath'] = os.path.join(paths['poseOutDir'], 'poses.yml')
	paths['cameraPath'] = os.path.join(paths['rgbDir'], 'camera.yml')
	return paths

def parse_3D_keypoints(keypointsPath, addCenter=False):
	# Returns 3*nKeypoints ndarray
	keypoints = np.loadtxt(keypointsPath, delimiter=',')
	if addCenter:
		zeroVector = np.zeros((3,1))
		if not np.any(np.all(keypoints==zeroVector, axis=0)):
			keypoints = np.hstack((np.zeros((3,1)), keypoints))
	return keypoints

def parse_pose(poseData, iView, iPose):
	R = np.array(poseData[iView][iPose]['cam_R_m2c']).reshape((3,3))
	t = np.array(poseData[iView][iPose]['cam_t_m2c']).reshape((3,1))
	pose = np.hstack((R, t))
	return pose

def parse_motion(motionPath):
	motion = np.loadtxt(motionPath)
	nViews = motion.shape[0]
	motion = motion.reshape((nViews, 3, 4), order='F')
	motion[:,:,0:3] = motion[:,:,0:3].transpose((0,2,1))
	return motion

def parse_inner_parameters(cameraPath):
	cameraData = yaml.load(open(cameraPath, 'r'))
	K = np.zeros((3,3))
	K[0,0] = cameraData['fx']
	K[1,1] = cameraData['fy']
	K[0,2] = cameraData['cx']
	K[1,2] = cameraData['cy']
	K[2,2] = 1
	return K