import os
import random
import numpy as np
import itertools
import yaml
from matplotlib import colors as mcolors
import cv2
from plyfile import PlyData
















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






































# USEFUL ONLY FOR THIS PROJECT
#############################################

def find_nbr_scenes(scenesDir):
	return len([x for x in os.listdir(scenesDir) if 
				os.path.isdir(os.path.join(scenesDir, x))]) #  & ('Scene' in x)

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
	classNameToIdx = {classNames[i]:(i+1) for i in range(0, len(classNames))}

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
	if className is not None:
		paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
		paths['keypointsPath'] = os.path.join(paths['modelDir'], str(classNameToIdx[className])+'_keypoints.txt')
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

def parse_motion(motionPath):
	motion = np.loadtxt(motionPath)
	nViews = motion.shape[0]
	motion = motion.reshape((nViews, 3, 4), order='F')
	motion[:,:,0:3] = motion[:,:,0:3].transpose((0,2,1))
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

def calculate_pose(points2D, points3D,camera_matrix):
	# Input: 
	# points2D - ndarray, shape=(nInstances,2,nKeypoints)
	# points3D - ndarray, shape=(3,nKeypoints)
	
	# Output:
	# poses - ndarray, shape=(nInstances,3,4)
	
	poses = np.zeros((len(points2D),3,4))
	for iPoint, point in enumerate(points2D):
		# Solve pnp problem
		poseParam = cv2.solvePnP(objectPoints = points3D.T, imagePoints = point.T[:,None],\
									 cameraMatrix = camera_matrix, distCoeffs = None,flags = cv2.SOLVEPNP_EPNP)
		
		# Extract pose related data and calculate pose
		R = cv2.Rodrigues(poseParam[1])[0]
		t = poseParam[2]
		pose = np.hstack((R,t))
		poses[iPoint] = pose
	return poses


def load_model_pointcloud(modelDir, modelIdx):
	modelPath = os.path.join(modelDir, str(modelIdx)+'.ply')
	plyData = PlyData.read(modelPath)
	vertex = plyData['vertex']
	(x,y,z) = (vertex[coord] for coord in ('x', 'y', 'z'))
	X = np.stack((x,y,z), axis=1)
	return X
