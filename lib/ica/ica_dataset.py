import os
import sys
print('iojijewoifjewfjfj')

# Own modules
from lib.ica.utils import * # Includes parse_3D_keypoints, pflat, pextend etc. 

# PVNet modueles
from lib.networks.model_repository import Resnet18_8s
from lib.utils.data_utils import read_rgb_np
from lib.datasets.linemod_dataset import compute_vertex_hcoords

# Other modules
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from numpy import zeros
from PIL import Image
from numpy import eye, unique
import yaml

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd

# Helper functions
def load_inner_parameters(sceneDirs):
	K = []
	for sceneDir in sceneDirs:
		cameraPath = os.path.join(sceneDir, 'rgb/camera.yml')
		thisK = parse_inner_parameters(cameraPath)
		K.append(thisK)
	return K
def load_pose_data(sceneDirs, verbose=False):
	poseData = []
	nScenes = len(sceneDirs)
	for iScene, sceneDir in enumerate(sceneDirs):
		if verbose:
			sys.stdout.write("\rReading pose data for scene {}/{}".format(iScene+1,nScenes))
			sys.stdout.flush()
		posesPath = os.path.join(sceneDir, 'poseannotation_out/poses.yml')
		thisPoseData = yaml.load(open(posesPath, 'r'), Loader=yaml.UnsafeLoader)
		poseData.append(thisPoseData)
	if verbose: sys.stdout.write('\n')
	return poseData
def load_visibility_data(visibilityPath):
	with open(visibilityPath, 'r') as f:
		visibilityData = yaml.load(f, Loader=yaml.UnsafeLoader)
	return visibilityData

# ICA Dataset Class (Complete with init, getitem and len)
class IcaDataset(Dataset):
	def __init__(self, classIdx, scenesDir, formats, keypoints, visibilityThreshold=None):

		# Trivial class member initilizations
		self.classIdx = classIdx # NOTE: Indexed from 1 to nClasses
		self.formats = formats # TO-DO: Deprecate
		self.keypoints = keypoints
		self.visibilityThreshold = visibilityThreshold

		# Create one list of absolute paths to specific scene directories. --- WARNING: DO NOT USE os.listdir() ON scenesDir AGAIN, AS IT MAY RETURN DIRECTORIES IN A RANDOM ORDER
		self.sceneDirs = [os.path.join(scenesDir, sceneDirName, 'pvnet') for sceneDirName in sorted(os.listdir(scenesDir))]

		# Load inner parameters (just choose the K matrix of the first scene)
		self.K = load_inner_parameters(self.sceneDirs)

		# Load poseData for each scene (list of length nScenes)
		self.poseData = load_pose_data(self.sceneDirs, verbose=True)

		# Define transformation to normalize input images with ImageNet values
		self.test_img_transforms = transforms.Compose([
			transforms.ToTensor(), # if image.dtype is np.uint8, then it will be divided by 255
			transforms.Normalize(mean=imageNetMean,
								 std=imageNetStd)
		])

		# Store formats for each individual scene in a list of length nScenes
		self.formatList = parse_rgb_formats(self.sceneDirs)

		# Check number of images across all Scene directories to determine self.len
		# Also build self.idxToSceneView, mapping sampler index to scene and viewpoint index
		skipInvisible = (self.visibilityThreshold is not None)
		self.len = 0
		self.idxToSceneView = []
		nScenes = len(self.sceneDirs)
		for iScene in range(nScenes):
			sceneDir = self.sceneDirs[iScene]
			thisRgbDir = os.path.join(sceneDir, 'rgb')
			thisSegDir = os.path.join(sceneDir, 'seg')
			thisNFiles = find_nbr_of_files(thisRgbDir, format=self.formatList[iScene])
			assert(thisNFiles == find_nbr_of_files(thisSegDir, format=self.formats['segFormat']))
			if skipInvisible:
				visibilityPath = os.path.join(sceneDir, 'visibility.yml')
				visibility = load_visibility_data(visibilityPath)
				instanceIdx = [iInstance for iInstance, instanceDict in enumerate(self.poseData[iScene][0]) if instanceDict['obj_id']==self.classIdx] # Assuming the same instances have entries for every viewpoint
				if instanceIdx == []:
					classVisibility = [0]*thisNFiles
				else:
					self.instanceVisibility = [[visibility[iImg][iInstance]['visibility'] for iInstance in instanceIdx] for iImg in range(thisNFiles)] #instanceVisibility[iImg][iInstance] is the visibility of instance iInstance in image iImg
					classVisibilityAlt = [max(self.instanceVisibility[iImg]) for iImg in range(thisNFiles)]
					classVisibility = [max([visibility[iImg][iInstance]['visibility'] for iInstance in instanceIdx]) for iImg in range(thisNFiles)] # List over best visibilities in all images in the scene
					assert(classVisibilityAlt==classVisibility)
				thisNVisibleFiles = sum(1 for vis in classVisibility if vis >= self.visibilityThreshold)
				self.len += thisNVisibleFiles
				self.idxToSceneView += [(iScene+1, iImg+1) for iImg in range(thisNFiles) if classVisibility[iImg] >= self.visibilityThreshold]
				print('Found ' + str(thisNVisibleFiles) + ' images in ' + thisRgbDir + ' where at least one instance is visible.')
			else:
				self.len += thisNFiles
				self.idxToSceneView += [(iScene+1,iImg+1) for iImg in range(thisNFiles)]
				print('Found ' + str(thisNFiles) + ' images in ' + thisRgbDir)

		# Assert conditions
		assert(self.len == len(self.idxToSceneView))

	def __getitem__(self, index):
		# OUTPUTS:
		# rgb - input rgb tensor [3, height, width]
		# mask - ground truth segmentation tensor [2, height, width]
		# ver - ground truth vertex tensor [2*nKeypoints, height, width]
		# verWeights - ground truth binary weight tensor [_, height, width]

		# Decide which scene and which image this sample index corresponds to
		sceneIdx, viewIdx = self.idxToSceneView[index]
		sceneDir = self.sceneDirs[sceneIdx-1]

		# RGB
		rgbPath = os.path.join(sceneDir, 'rgb', str(viewIdx).zfill(self.formats['rgbNLeadingZeros'])+'.'+self.formatList[sceneIdx-1])
		rgb = read_rgb_np(rgbPath)
		rgb = self.test_img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))
		_, height, width = rgb.shape

		# Mask
		segDir = os.path.join(sceneDir, 'seg')
		segPath = os.path.join(segDir, str(viewIdx).zfill(self.formats['segNLeadingZeros'])+'.'+self.formats['segFormat'])
		segImg = np.asarray(Image.open(segPath))
		instanceIdxPath = os.path.join(segDir, 'classindex.txt')
		instanceIdx = np.loadtxt(instanceIdxPath, delimiter=',')
		if instanceIdx.shape == (): # Special case for single instance
			instanceIdx.shape = (1,)

		idxMatch = (instanceIdx == self.classIdx).astype(int)
		idxMatch = np.concatenate(([0], idxMatch)) # Add 'no-class' idx
		mask = torch.tensor(idxMatch[segImg], dtype=torch.int64)

		# Vertex
		nKeypoints = (self.keypoints).shape[1]
		ver = np.zeros([height,width,nKeypoints*2],np.float32)
		instanceSegImg = mask.numpy()*segImg
		nInstances = sum(idxMatch)

		idx = [i for i,j in enumerate(instanceIdx) if j==self.classIdx] 
		poses = [parse_pose(self.poseData[sceneIdx-1], viewIdx-1, iPose) for iPose in idx]

		for iInstance in range(nInstances):
			thisMask = instanceSegImg == idx[iInstance]+1
			keypointsProjected = self.K[sceneIdx-1] @ (poses[iInstance] @ pextend(self.keypoints, vecType='col'))
			keypointsProjected = pflat(keypointsProjected)
			ver = ver + compute_vertex_hcoords(thisMask, keypointsProjected.T)

		ver=torch.tensor(ver, dtype=torch.float32).permute(2, 0, 1)
		verWeights=mask.unsqueeze(0).float()

		return rgb, mask, ver, verWeights


	# Temporary shit
	def get_2d_points(self, index):
		# OUTPUTS:
		# rgb - input rgb tensor [3, height, width]
		# mask - ground truth segmentation tensor [2, height, width]
		# ver - ground truth vertex tensor [2*nKeypoints, height, width]
		# verWeights - ground truth binary weight tensor [_, height, width]

		# Decide which scene and which image this sample index corresponds to
		sceneIdx, viewIdx = self.idxToSceneView[index]
		sceneDir = self.sceneDirs[sceneIdx-1]

		# Load poses
		segDir = os.path.join(sceneDir, 'seg')
		instanceIdxPath = os.path.join(segDir, 'classindex.txt')
		instanceIdx = np.loadtxt(instanceIdxPath, delimiter=',')
		if instanceIdx.shape == (): # Special case for single instance
			instanceIdx.shape = (1,)
		idx = [i for i,j in enumerate(instanceIdx) if j==self.classIdx] 
		nInstances = len(idx)
		poses = [parse_pose(self.poseData[sceneIdx-1], viewIdx-1, iPose) for iPose in idx]

		# Calculate 2D points
		nKeypoints = (self.keypoints).shape[1]
		visibleInstanceIdx = [idx for idx,visibility in enumerate(self.instanceVisibility[viewIdx-1]) if visibility > self.visibilityThreshold]
		nInstances = len(visibleInstanceIdx)
		assert(nInstances > 0)

		visiblePoses = np.zeros((nInstances, 3, 4))
		visiblePoints2D = np.zeros((nInstances, nKeypoints, 2))
		for iDet, iInstance in enumerate(visibleInstanceIdx):
			visiblePoses[iDet] = poses[iInstance]
			keypointsProjected = self.K[sceneIdx-1] @ (visiblePoses[iDet] @ pextend(self.keypoints, vecType='col'))
			keypointsProjected = pflat(keypointsProjected)
			visiblePoints2D[iDet] = keypointsProjected.transpose()[:,0:2]

		return visiblePoints2D, visiblePoses

	def getIdxSceneViewMap(self):
		return self.idxToSceneView

	def __len__(self):
		return self.len