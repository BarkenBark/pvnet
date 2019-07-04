import os
import sys
import time
import pickle
import yaml
sys.path.append('.')
sys.path.append('..')

from lib.ica.utils import * # Includes parse_3D_keypoints, etc. 

from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
from PIL import Image

motion = parse_motion('/var/www/webdav/Data/ICA/Scenes/Validation/Scene13/poseannotation/output/motionScaled.txt')
instancesIdx = pickleLoad('/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10/instancesIdx.pickle')
points2DInstance = pickleLoad('/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10/points2DInstance.pickle')
modelKeypoints = parse_3D_keypoints('/var/www/webdav/Data/ICA/Scenes/Validation/Scene13/poseannotation/models/10_keypoints.txt', addCenter=True).T
cameraMatrix = parse_inner_parameters('/var/www/webdav/Data/ICA/Scenes/Validation/Scene13/poseannotation/rgb/camera.yml')

class ImageKeypointGenerator2(AxesGenerator):
	def __init__(self, imgDir, imgFormats, points, instancesIdx):
		# Input:
		# imgDir - directory containing N images
		# imgFormats - dictionary with keys 'format' (e.g. 'png', 'jpg') and 'nLeadingZeros' (e.g. 5)
		# points - list of length N where element i contains an (nInstances, nKeypoints, 2) np array
		self.imgDir = imgDir
		self.format = imgFormats['format']
		self.nLeadingZeros = imgFormats['nLeadingZeros']
		self.points = points
		self.instancesIdx = instancesIdx
		self.len = len(self.instancesIdx)
		assert(len(points[0])==self.len)

	def __update_ax__(self, ax, index):
		index = index % self.len
		index = instancesIdx[index]
		imgPath = os.path.join(self.imgDir, str(index+1).zfill(self.nLeadingZeros)+'.'+self.format)
		img = self.read_img_np(imgPath)
		ax.clear()
		ax.imshow(img)
		colorCycler = pltColorCycler()
		for instancePoints in self.points:
			color = next(colorCycler)
			x = instancePoints[index]
			print(x)
			if x is not None:
				ax.scatter(x[:,0], x[:,1], c=color)
			ax.set_title("Viewpoint {}".format(index))

	def __len__(self):
		return self.len

	def read_img_np(self, imgPath):
		img = Image.open(imgPath).convert('RGB')
		img = np.array(img,np.uint8)
		return img


class GtPoseGenerator(AxesGenerator):
	def __init__(self, imgDir, formats, instancesIdx, poseInEye, motion, keypoints3D, K):
		# Input:
		# imgDir - directory containing N images
		self.imgDir = imgDir
		self.formats = formats
		self.instancesIdx = instancesIdx
		self.poseInEye = poseInEye
		self.motion = motion
		self.covariance = covariance
		self.keypoints3D = keypoints3D
		self.K = K
		self.len = len(instancesIdx)
		assert(self.len==len(motion))

	def __update_ax__(self, ax, index):
		print('asasdasdasdasd')
		index = index % self.len
		index = self.instancesIdx[index]

		# Clear axes
		ax.clear()
		ax.set_title("Viewpoint {}".format(index))

		# Load image
		imgPath = os.path.join(self.imgDir, str(index+1).zfill(self.formats['rgbNLeadingZeros'])+'.'+self.formats['rgbFormat'])
		img = self.read_img_np(imgPath)
		height, width, _ = img.shape
		ax.imshow(img)

		# Calculate points and plot them 
		thisCamera = self.motion[index]
		pose = thisCamera@self.poseInEye
		x = pflat(pose@pextend(self.keypoints3D))
		ax.scatter(x[:,0], x[:,1])


		# ax.set_xlim(0, width)
		# ax.set_ylim(height, 0)



	def __len__(self):
		return self.len

	def read_img_np(self, imgPath):
		img = Image.open(imgPath).convert('RGB')
		img = np.array(img,np.uint8)
		return img

def eulerAnglesToRotationMatrix(theta) :
	#theta = theta[::-1]
	 
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

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-3
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
	#assert(isRotationMatrix(R))
	 
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
	return np.array([x,y,z])



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
	print('pose0:  ',pose0)
	t0 = pose0[:,3]
	R0 = pose0[:,0:3]
	eulAngle0 = rotationMatrixToEulerAngles(R0)
	eulPose0 = np.concatenate((eulAngle0,t0))
	
	sol=scipy.optimize.minimize(lambda eulPose: reprojection_error(imagePointsNormalizedList, modelPointsList, eulPose, cameras), eulPose0,tol=0.0000001,method='Powell')
	eulImproved = sol.x
	
	RPose = eulerAnglesToRotationMatrix(eulImproved[0:3])
	tPose = eulImproved[3:6]
	improvedPose = np.hstack((RPose,tPose[:,None]))
	return improvedPose


# Load paths
resultsDirs = '/var/www/webdav/Data/ICA/Results/Experiment4'
sceneDirs = '/var/www/webdav/Data/ICA/Scenes/Validation'
iScene = 13
iClass = 10
formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0, 'segFormat':'png', 'segNLeadingZeros':5}

# Implicit
resultsDir = os.path.join(resultsDirs, 'Scene'+str(iScene), 'Class'+str(iClass))
sceneDir = os.path.join(sceneDirs, 'Scene'+str(iScene), 'pvnet')
paths = get_work_paths(sceneDir)
predSegDir = os.path.join(resultsDir, 'instanceMasks')
nViews = find_nbr_of_files(paths['rgbDir'], formats['rgbFormat'])




#keepIdx = [idx for idx, point in enumerate(keypointsList) if point is not None] # Remove points2D at indices where None
#keypointsList = [keypointsList[idx] for idx in keepIdx]
poseData = yaml.load(open(paths['posesPath'], 'r'), Loader=yaml.UnsafeLoader)
poseInEye = parse_pose(poseData, 0, 9)
classNameToIdx, classIdxToName = create_class_idx_dict(paths['modelDir'])
pathsClass = get_work_paths(sceneDir, classIdxToName[iClass], classNameToIdx)
keypoints3D = parse_3D_keypoints(pathsClass['keypointsPath'], addCenter=True)
K = parse_inner_parameters(paths['cameraPath'])
motion = [motion[idx] for idx in instancesIdx]

# Check that the instances are indeed consistent
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
rgbFormats = {'format':formats['rgbFormat'], 'nLeadingZeros':formats['rgbNLeadingZeros']}
axGen = ImageKeypointGenerator2(paths['rgbDir'], rgbFormats, points2DInstance, instancesIdx)
plotScroller = PlotScroller(ax1, axGen)
cid = fig1.canvas.mpl_connect('scroll_event', plotScroller.onscroll)
plt.show()


# Select an instance, and filter indices where points are None
iInstance = 1
keypointsList = points2DInstance[iInstance]
keepIdx = [idx for idx, point in enumerate(keypointsList) if point is not None] # Remove points2D at indices where None
instancesIdxKeep = [instancesIdx[idx] for idx in keepIdx]
keypointsList = [keypointsList[idx] for idx in keepIdx]
cameraTmp = [motion[idx] for idx in keepIdx]
modelPointsList = [modelKeypoints for idx in keepIdx]


# Check that the motion actually corresponds to the instance for one instance
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
rgbFormats = {'format':formats['rgbFormat'], 'nLeadingZeros':formats['rgbNLeadingZeros']}
axGen = GtPoseGenerator(paths['rgbDir'], rgbFormats, instancesIdxKeep, poseInEye, cameraTmp, keypoints3D, K)
plotScroller = PlotScroller(ax1, axGen)
cid = fig2.canvas.mpl_connect('scroll_event', plotScroller.onscroll)
plt.show()


improvedPose = multiviewPoseEstimation(keypointsList, modelPointsList, cameraTmp, cameraMatrix,pose0=np.array([[-0.0804, 0.9961, 0.0358, 0.3045],[0.2041, 0.0516, -0.9776, -0.0278],[-0.9756, -0.0713, -0.2075, 0.5793]]))


