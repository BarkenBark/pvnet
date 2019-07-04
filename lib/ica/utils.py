import os
import time
import random
import numpy as np
import math
import itertools
import pickle
import yaml
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cv2
from open3d import read_point_cloud
import torch
from lib.utils.extend_utils.extend_utils import uncertainty_pnp
# from bbox import BBox3D
# from bbox.metrics import iou_3d
from scipy.spatial.distance import pdist
import scipy














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
# TO-DO: 
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


# Return every 'nSkips' datapoints for all objects in args.
# Requires the length of the objects in args to have same size along first dimension/axis
def skip_down_sample(nViewSkips=1, *args):

	nElements = len(args[0])
	for iArg in len(args):
		assert(len(args[iArg])==nElements)
	
	# Set how many wiews to skip, i.e. use every 'nViewSkips' cameras
	selectedElementsRange = range(0,nElements,nViewSkips)
	selectedElementsSlice = slice(0,nElements,nViewSkips)
	selectedData = tuple([arg[selectedElementsSlice] if type(arg) == list else arg[selectedElementsRange] for arg in args])
	return selectedData

def get_selected_data(idx, *args):
	nElements = len(args[0])
	for arg in args:
		assert(len(arg)==nElements)
	selectedData = tuple([[arg[i] for i in idx] for arg in args])
	return selectedData

def changeNumFormat(directory, fileFormat=None, nLeadingZeros=5):
	for fileName in os.listdir(directory):
		if fileFormat is not None:
			if fileName.endswith(fileFormat):
				fileNameNoExt = os.path.splitext(fileName)[0]
				thisFileFormat = os.path.splitext(fileName)[1]
				try:
					num = int(fileNameNoExt)
					newFileName = str(num).zfill(nLeadingZeros)+thisFileFormat
					os.rename(fileName, newFileName)
				except ValueError:
					pass

def pickleSave(var, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as file:
		pickle.dump(var, file)

def pickleLoad(path):
	with open(path, 'rb') as file:
		return pickle.load(file)

class AxesGenerator(object):
    # Abstract class
    # Subclasses should reutrn a pltpyplot-figure object when calling __getfig__()
    def __update_ax__(self, ax, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

# Updates the axes object of a figure when you scroll according to an Axes-generator which outputs an axes according to an index 
class PlotScroller(object):
    def __init__(self, ax, axGen):
        # Axes Generator
        self.axGen = axGen # Instance of AxesGenerator
        self.len = self.axGen.len
        self.ax = ax
        self.ax.set_title('use scroll wheel to navigate images')

		# State
        self.index = 0

        # Initialize
        self.draw()

    def onscroll(self, event): # Callback function to be called with fig.canvas.mpl_connect('scroll_event', thisFunction)
        if event.button == 'up':
            self.index = (self.index + 1) % self.len
        else:
            self.index = (self.index - 1) % self.len
        self.draw()

    def draw(self):
        self.axGen.__update_ax__(self.ax, self.index)
        plt.draw()


def keep_every_nth(N, n):
	if n > 0:
		idxList = [idx*n+1 for idx in range(N) if idx < N/n]
	else:
		idxList = [idx+1 for idx in range(N)]
	return idxList





























# USEFUL FOR COMPUTER VISION
###################################################################

def pflat(x):
	return x/x[-1,:]

def pextend(x, vecType='row'):
	if len(x.shape) == 1: # Single point case
		return np.insert(x, len(x), 1)
	elif len(x.shape) == 2: 
		if vecType=='row':
			onePad = np.ones((x.shape[0],1))
			return np.hstack((x,onePad))
		elif vecType=='col':
			onePad = np.ones((1,x.shape[1]))
			return np.vstack((x,onePad))
	else:
		print('The fuck you trying to do, returning NoneType')

def crossmat(x):
	return np.array([[0,-x[2],x[1]], [x[2],0,-x[0]], [-x[1],x[0],0]])

def camera_center(P):
	R = P[0:3,0:3]
	t = P[:,3]
	C = (-R.transpose()@t).reshape((3,1))
	return C

def normalize_2d(x, K):
	# Assume x is [nPoints, 2] (projection)
	xn = np.linalg.inv(K)@np.transpose(np.insert(x, 2, 1, axis=1))
	xn = np.transpose(xn)[:,0:2]
	return xn

# Project 3D points X (nPoints, 3) in local coordinate system using camera P (3, 4) 
def project(X, P):
	x = pflat(P@pextend(X, vecType='col'))
	x = x[0:2,:]
	x = x.T
	return x

def compute_viewing_ray(camera, point):
	# camera - [3,4] numpy array
	# point - numpy array of shape (2,)
	#location = -camera[0:3,0:3]@camera[:,3]
	location = camera_center(camera)
	direction = camera[0:3,0:3].transpose()@pextend(point)
	ray = np.hstack((location.reshape(3,1), direction.reshape(3,1)))
	return ray

def normalize_list(points, K):
	pointsNormalized = [] # To be filled with nCameras arrays of shape [nInstances, nKeypoints, 2]
	nCameras = len(points)
	for iCam in range(nCameras):
		thisPoints = points[iCam]
		if thisPoints is None:
			pointsNormalized.append(None)
			continue
		nInstances = thisPoints.shape[0]
		thisPointsNormalized = np.zeros(thisPoints.shape)
		for iInstance in range(nInstances):
			x = thisPoints[iInstance]
			thisPointsNormalized[iInstance] = normalize_2d(x, K)
		pointsNormalized.append(thisPointsNormalized)
	return pointsNormalized

def epipolar_lines_list_to_array(epiLines):
	epiLinesArray = np.empty((3,0))
	for i, lines in enumerate(epiLines):
		if lines is not None:
			epiLinesArray = np.append(epiLinesArray, lines, axis=1)


def get_centers(poses):
	# detections.shape==(b,3,4) or (3,4) where b is batch size
	# If batch size > 1, calculate center with tensor multiplication
	
	if len(poses.shape) == 3:
		centers = poses[:,:,-1]
	else:
		centers = poses[:,-1]
	
# =============================================================================
#     if detections.shape[0] > 1 and len(detections.shape) == 3:
#         detections = np.swapaxes(detections,0,2)
#         centers = np.sum(detections[0:3,:,:] * detections[-1:,:,:],axis=1)
#     else:
#         centers = detections.squeeze()[:,0:3] @ detections.squeeze()[:,-1:]
# =============================================================================
	
	return centers

def eulerAnglesToRotationMatrix(theta) :
	R_x = np.array([[1,         0,                  0                   ],
					[0,         math.cos(theta[0]), -math.sin(theta[0]) ],
					[0,         math.sin(theta[0]), math.cos(theta[0])  ]
					])	 
	R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
					[0,                     1,      0                   ],
					[-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
					])
				 
	R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
					[math.sin(theta[2]),    math.cos(theta[2]),     0],
					[0,                     0,                      1]
					])
	R = np.dot(R_z, np.dot( R_y, R_x ))

	return R



def rotationMatrixToEulerAngles(R) :
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
 
	return np.array([x, y, z])

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

def add_metric(pose1, pose2, points):
	# pose is [3,4]
	# points in [nPoints, 3]
	add = np.mean(np.linalg.norm(pflat(pose1@pextend(points.T, 'col')) - pflat(pose2@pextend(points.T, 'col')), axis=0))
	return add


def reprojection_error(imagePointsList, modelPointsList, eulPose, cameras):
	#cameras: list of cameras
	#imagePointsList: list of length nViews, containing arrays of varying shape = (nImagePointsx2)
	#modelPointsList: list of length nViews, containing arrays of varying shape = (nImagePointsx2)
	#eulpose: (6,) pose where rotationmatrix is replaced with euler angles
	nViews = np.size(cameras,axis=0)
	RPose = eulerAnglesToRotationMatrix(eulPose[0:3])
	tPose = eulPose[3:6]
	reprojectionError = 0
	
	for iView in range (nViews):
		P = cameras[iView]
		R = P[:,0:3]
		t = P[:,3]
		imagePoints = imagePointsList[iView].T
		nImagePoints = imagePoints.shape[1]
		modelPoints = modelPointsList[iView].T
		for iImagePoint in range(nImagePoints):
			proj = R @ (RPose @ modelPoints[:,iImagePoint:iImagePoint+1]+tPose[:,None])+t[:,None]
			reprojectionError = reprojectionError + np.linalg.norm( [imagePoints[0,iImagePoint] - proj[0]/proj[2], imagePoints[1,iImagePoint] - proj[1]/proj[2] ])**2
			
	return reprojectionError


def reprojection_error_fast(imagePointsList, modelPointsList, eulPose, cameras):
    #cameras: list of cameras
    #imagePointsList: list of length nViews, containing arrays of varying shape = (nImagePointsx2)
    #modelPointsList: list of length nViews, containing arrays of varying shape = (nImagePointsx2)
    #eulpose: (6,) pose where rotationmatrix is replaced with euler angles
    nViews = np.size(cameras,axis=0)
    RPose = eulerAnglesToRotationMatrix(eulPose[0:3])
    tPose = eulPose[3:6]
    reprojectionError = 0
    
    for iView in range (nViews):
        P = cameras[iView]
        R = P[:,0:3]
        t = P[:,3]
        imagePoints = imagePointsList[iView].T
        nImagePoints = imagePoints.shape[1]
        modelPoints = modelPointsList[iView].T
        proj = R @ (RPose @ modelPoints+tPose[:,None])+t[:,None]
        reprojectionError = reprojectionError + np.linalg.norm(imagePoints-pflat(proj)[0:2])**2
            
    return reprojectionError


# =============================================================================
# imagePointsList = [np.array([[-0.0637,0.2170,1],[-0.1002,0.2270,1],[-0.1522,0.0946,1],[-0.1832,0.0606,1],[-0.1747,0.1396,1],[-0.1252,0.0636,1],[-0.1207,0.1171,1]]),np.array([[0.4195,0.0093,1],[0.3801,0.0043,1],[0.4502,-0.2242,1],[0.4519,-0.1535,1],[0.5106,-0.2116,1],[0.5119,-0.1632,1]])]
# modelPointsList = [np.array([[-0.0161,0.0437,-0.0867],[0.0153,0.0438,-0.0850],[-0.0009,-0.0017,0.1114],[0.0135,-0.0224,0.1100],[0.0141,0.0199,0.1108],[-0.0175,-0.0169,0.1104],[-0.0176,0.0125,0.1109]]),np.array([[-0.0161,0.0437,-0.0867],[0.0153,0.0438,-0.0850],[0.0135,-0.0224,0.1100],[0.0141,0.0199,0.1108],[-0.0175,-0.0169,0.1104],[-0.0176,0.0125,0.1109]])]
# cameras = [np.array([[0.6763,0.3008,-0.6724,0.7985],[-0.7087,0.5145,-0.4827,0.2955],[0.2008,0.8030,0.5611,0.2399]]),
#            np.array([[0.8411,0.3174,-0.4380,0.9958],[-0.5391,0.5570,-0.6317,0.3421],[0.0434,0.7675,0.6396,0.1189]])]
# eulPose = np.array([1,2,3,0,0,1])
# 
# =============================================================================
	
def multiviewPoseEstimation(imagePointsList,modelPointsList,cameras,cameraMatrix,pose0=None):
	Kinv = np.linalg.inv(cameraMatrix)
	imagePointsNormalizedList = [(Kinv[:2,:] @ pextend(imagePoints.T,'col')).T for imagePoints in imagePointsList]
	if pose0 is None:
		imagePoints = imagePointsList[0]
		modelPoints = modelPointsList[0]
		camera = cameras[0]
		poseParam = cv2.solvePnP(objectPoints = modelPoints, imagePoints = imagePoints[:,None,0:2],\
									 cameraMatrix = cameraMatrix,distCoeffs=None, flags = cv2.SOLVEPNP_EPNP)
		R = cv2.Rodrigues(poseParam[1])[0]
		t = poseParam[2]
		pose = np.hstack((R,t))
		firstCamera = np.hstack((np.eye(3),np.zeros((3,1))))
		pose0 = transform_pose(pose, camera, firstCamera).squeeze()
	#print('pose0:  ',pose0)
	t0 = pose0[:,3]
	R0 = pose0[:,0:3]
	eulAngle0 = rotationMatrixToEulerAngles(R0)
	eulPose0 = np.concatenate((eulAngle0,t0))
	
	#sol=scipy.optimize.minimize(lambda eulPose: reprojection_error(imagePointsNormalizedList, modelPointsList, eulPose, cameras), eulPose0,tol=0.0000001,method='Powell')
	sol=scipy.optimize.minimize(lambda eulPose: reprojection_error_fast(imagePointsNormalizedList, modelPointsList, eulPose, cameras), eulPose0,tol=0.0000001,method='Powell')
	eulImproved = sol.x
	
	RPose = eulerAnglesToRotationMatrix(eulImproved[0:3])
	tPose = eulImproved[3:6]
	improvedPose = np.hstack((RPose,tPose[:,None]))
	return improvedPose





























# USEFUL ONLY FOR THIS PROJECT
#############################################

def find_nbr_scenes(scenesDir):
	return len([x for x in os.listdir(scenesDir) if 
				os.path.isdir(os.path.join(scenesDir, x))]) #  & ('Scene' in x)

def get_scene_name(sceneDir):
	# Input: path/to/sceneName/pvnet
	# Output sceneName
	return list(filter(None, sceneDir.split(os.path.sep)))[-2] # Extracts second last element from path string

def create_class_idx_dict(modelDir):
	# Checks the names of the .ply files in the model directory (alt. model_names.txt), and creates a dictionary mapping modelnames to class indices

	# Check if model_names.txt exists
	modelNamesPath = os.path.join(modelDir, 'model_names.txt')
	nameListExists = os.path.isfile(modelNamesPath)

	# If model_names.txt exists, use class names as defined in the file (assume filenames are 1.ply, 2.ply, ..., nClasses.ply)
	# Otherwise, just use the filenames
	if nameListExists:
		with open(modelNamesPath, 'r') as file:
			classNames = [modelName.rstrip('\n') for modelName in file]
	else:
		filenames = os.listdir(modelDir)
		classNames = [s.replace('.ply', '') for s in filenames if s.endswith('.ply')]
		classNames.sort()
	classNameToIdx = {classNames[i]:(i+1) for i in range(len(classNames))}

	classIdxToName = {value:key for (key, value) in classNameToIdx.items()}

	return classNameToIdx, classIdxToName



def get_work_paths(dataDir, className=None, classNameToIdx=None):
	paths = dict()
	paths['dataDir'] = dataDir
	paths['networkDir'] = os.path.join(dataDir, 'network')
	paths['modelDir'] = os.path.join(dataDir, 'models')
	paths['poseOutDir'] = os.path.join(dataDir, 'poseannotation_out')
	paths['rgbDir'] = os.path.join(dataDir, 'rgb')
	paths['segDir'] = os.path.join(dataDir, 'seg')
	paths['posesPath'] = os.path.join(paths['poseOutDir'], 'poses.yml')
	paths['cameraPath'] = os.path.join(paths['rgbDir'], 'camera.yml')
	paths['motionPath'] = os.path.join(paths['poseOutDir'], 'motionScaled.txt')
	if className is not None:
		paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
		paths['keypointsPath'] = os.path.join(paths['modelDir'], str(classNameToIdx[className])+'_keypoints.txt')
		paths['className'] = className
	return paths

def parse_rgb_formats(sceneDirs):
	rgb_formats = []
	for sceneDir in sceneDirs:
		rgbDir = os.path.join(sceneDir, 'rgb')
		dirContent = os.listdir(rgbDir)
		if any([file.endswith('.jpg') for file in dirContent]):
			rgb_formats.append('jpg')
		elif any([file.endswith('.png') for file in dirContent]):
			rgb_formats.append('png')
		else:
			print('fudge to self')
			exit()

	# for iScene in range(1,nScenes+1):
	# 	dirContent = os.listdir(scenesDir+'/Scene'+str(iScene)+'/pvnet/rgb')
	# 	if any([file.endswith('.jpg') for file in dirContent]):
	# 		rgb_formats.append('jpg')
	# 	elif any([file.endswith('.png') for file in dirContent]):
	# 		rgb_formats.append('png')
	# 	else:
	# 		print('fudge to self')
	# 		exit()

	return rgb_formats


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

def parse_motion(motionPath, transposeR=False):
	motion = np.loadtxt(motionPath)
	nViews = motion.shape[0]
	motion = motion.reshape((nViews, 3, 4), order='F')
	if transposeR:
		motion[:,:,0:3] = motion[:,:,0:3].transpose((0,2,1)) # Fuck this line in particular
	return motion

def parse_inner_parameters(cameraPath):
	cameraData = yaml.load(open(cameraPath, 'r'), Loader=yaml.UnsafeLoader)
	K = np.zeros((3,3))
	K[0,0] = cameraData['fx']
	K[1,1] = cameraData['fy']
	K[0,2] = cameraData['cx']
	K[1,2] = cameraData['cy']
	K[2,2] = 1
	return K

def transform_pose(poseInCameraX, cameraX, cameraY):
	# Ouput: poseInCameraY - ndarray, shape = (3,4)
	poseTransform = np.vstack((poseInCameraX ,np.array([0, 0, 0, 1])[None,:]))

	RCam = cameraX[:,0:3]
	tCam = cameraX[:,-1:]
	PCam = np.hstack((RCam.T,-RCam.T @ tCam))

	camTransform  = np.vstack((PCam,np.array([0, 0, 0, 1])[None,:]))

	poseInCameraY = cameraY @ camTransform @ poseTransform
	return poseInCameraY




#############################################
# Temporary removoval due to bug with open3D
# Cant import open3D and torch at same time
#############################################
def load_model_pointcloud(modelDir, modelIdx, downsampling=None):
	modelPath = os.path.join(modelDir, str(modelIdx)+'.ply')
	pc = read_point_cloud(modelPath)
	X = np.transpose(np.asarray(pc.points)) #[3,nPoints]
	minBounds = pc.get_min_bound()
	maxBounds = pc.get_max_bound()
	geoCenter = np.mean(X, axis=1)
	if downsampling is not None:
		idx = np.random.permutation(X.shape[1])
		np.delete(X, np.arange(0, int(X.shape[1]*downsampling)), axis=1)
	return X, geoCenter, minBounds, maxBounds

def get_model_limits(modelDir, modelIdx):
	modelPath = os.path.join(modelDir, str(modelIdx)+'.ply')
	minBounds = read_point_cloud(modelPath).get_min_bound()
	maxBounds = read_point_cloud(modelPath).get_max_bound()
	return minBounds, maxBounds

def get_model_diameter(modelDir, modelIdx):
	#modelPath = os.path.join(modelDir, str(modelIdx)+'.ply')
	# X = read_point_cloud(modelPath).points
	# d = pdist(X)
	# return max(d)
	diameterPath = os.path.join(modelDir, str(modelIdx)+'_diameter.txt')
	d = np.asscalar(np.loadtxt(diameterPath))
	return d

# Convert from (nInstances, height, width) binary tensor to (height width) int tensor
def convertInstanceSeg(instanceSeg):
	w = np.arange(instanceSeg.shape[0]).astype('uint8')+1
	w = w[:,None,None]
	newInstanceSeg = np.sum(instanceSeg*w, axis=0)
	return newInstanceSeg

def segpred_to_mask(segPred):
	return np.argmax(segPred, axis=1).astype('uint8')

def vertexfield_to_angles(verPred, mask, keypointIdx=0):
	print(mask.shape)
	_,h,w=mask.shape
	angle = np.zeros((h,w))
	x,y = verPred[0,[2*keypointIdx, 2*keypointIdx+1],:,:]
	angle = np.arctan2(-y,x)
	angle = angle*mask[0,:,:]
	return angle

# def pose_to_bounding_box(pose, modelDir, modelIdx):
# 	R = pose[:3,:3]
# 	t = pose[:,3]
# 	eul = rotationMatrixToEulerAngles(R)
# 	eul = list(eul)
# 	minBounds, maxBounds = get_model_limits(modelDir, modelIdx)
# 	l = maxBounds[0]-minBounds[0]
# 	w = maxBounds[1]-minBounds[1]
# 	h = maxBounds[2]-minBounds[2]
# 	bbox = BBox3D(t[0], t[1], t[2], l, w, h, euler_angles=eul)
# 	return bbox

# def iou(bbox1, bbox2):
# 	return iou_3d(bbox1, bbox2)

# poseInEyeEst = np.array([[-0.0804, 0.9961, 0.0358, 0.3045],[0.2041, 0.0516, -0.9776, -0.0278],[-0.9756, -0.0713, -0.2075, 0.5793]])

