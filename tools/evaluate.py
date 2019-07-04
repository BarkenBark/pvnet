# Script to evaluate

# Import statements
import os
import sys
sys.path.append('.')
sys.path.append('..')

# Other modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
import itertools


# Own modules
from lib.ica.utils import *
from lib.ica.run_utils import covar_to_weight

# PVNet modules
from lib.utils.data_utils import read_rgb_np
from lib.utils.evaluation_utils import pnp
from lib.utils.extend_utils.extend_utils import uncertainty_pnp



# DEFINE CLASSES
#############################################

# Tool for plotting ad scrolling through images with keypoints plotted in it
class ImageKeypointGenerator(AxesGenerator):
	def __init__(self, imgDir, imgFormats, points, imgIdx=None):
		# Input:
		# imgDir - directory containing N images
		# imgFormats - dictionary with keys 'format' (e.g. 'png', 'jpg') and 'nLeadingZeros' (e.g. 5)
		# points - list of length N where element i contains an (nInstances, nKeypoints, 2) np array
		self.imgDir = imgDir
		self.format = imgFormats['format']
		self.nLeadingZeros = imgFormats['nLeadingZeros']
		self.points = points
		self.imgIdx = imgIdx

		nImages = find_nbr_of_files(self.imgDir, self.format)
		if self.imgIdx is None:
			self.imgIdx = list(range(nImages))

		self.len = len(imgIdx)
		assert(len(points)==self.len)

	def __update_ax__(self, ax, index):
		index = index % self.len
		index = self.imgIdx[index]

		imgPath = os.path.join(self.imgDir, str(index+1).zfill(self.nLeadingZeros)+'.'+self.format)
		img = self.read_img_np(imgPath)
		ax.clear()
		ax.imshow(img)
		x = self.points[index]
		if x is not None:
			nInstances = x.shape[0]
			for iInstance in range(nInstances):
				ax.scatter(x[iInstance,:,0], x[iInstance,:,1])
		ax.set_title("Viewpoint {}".format(index))

	def __len__(self):
		return self.len

	def read_img_np(self, imgPath):
		img = Image.open(imgPath).convert('RGB')
		img = np.array(img,np.uint8)
		return img


#pltColorsBase = list(dict(mcolors.BASE_COLORS).keys())
# Tool for generating segmentation overlap axes
class SegmentationOverlapGenerator(AxesGenerator):
	def __init__(self, imgDir, segDir, formats):
		# Input:
		# imgDir - directory containing N images
		self.imgDir = imgDir
		self.segDir = segDir
		self.formats = formats
		self.len = find_nbr_of_files(imgDir, self.formats['rgbFormat'])
		assert(self.len==find_nbr_of_files(segDir, self.formats['segFormat']))

	def __update_ax__(self, ax, index):
		index = index % self.len

		imgPath = os.path.join(self.imgDir, str(index+1).zfill(self.formats['rgbNLeadingZeros'])+'.'+self.formats['rgbFormat'])
		img = self.read_img_np(imgPath)

		segPath = os.path.join(self.segDir, str(index+1).zfill(formats['segNLeadingZeros'])+'.'+self.formats['segFormat'])
		seg = np.asarray(Image.open(segPath))
		nInstances = np.max(seg)

		imgOverlap = img.copy()
		colorCycler = pltColorCycler()
		for iInstance in range(1,nInstances+1):
			colorString = next(colorCycler)
			color = np.array(mcolors.BASE_COLORS[colorString])*128
			imgOverlap[seg==iInstance] = img[seg==iInstance]//2 + color[None,None,:] #+ colorArr[iInstance-1,:][None,None,:]
		ax.clear()
		ax.imshow(imgOverlap)
		ax.set_title("Viewpoint {}".format(index))

	def __len__(self):
		return self.len

	def read_img_np(self, imgPath):
		img = Image.open(imgPath).convert('RGB')
		img = np.array(img,np.uint8)
		return img


class SegKeypointsCovarianceGenerator(AxesGenerator):
	def __init__(self, imgDir, segDir, formats, points, covariance, imgIdx=None):
		# Input:
		# imgDir - directory containing N images
		self.imgDir = imgDir
		self.segDir = segDir
		self.formats = formats
		self.points = points
		self.covariance=covariance
		self.imgIdx = imgIdx

		nImages = find_nbr_of_files(self.imgDir, self.formats['rgbFormat'])
		if self.imgIdx is None:
			self.imgIdx = list(range(nImages))


		self.len = len(self.imgIdx)
		#assert(find_nbr_of_files(imgDir, self.formats['rgbFormat'])==find_nbr_of_files(segDir, self.formats['segFormat']))
		assert(self.len==len(points))
		assert(self.len==len(covariance))

	def __update_ax__(self, ax, index):
		index = index % self.len
		imgIndex = self.imgIdx[index]-1

		# Clear axes
		ax.clear()
		ax.set_title("Viewpoint {}".format(imgIndex))

		# Load image
		imgPath = os.path.join(self.imgDir, str(imgIndex+1).zfill(self.formats['rgbNLeadingZeros'])+'.'+self.formats['rgbFormat'])
		img = self.read_img_np(imgPath)

		# Load segmentation
		segPath = os.path.join(self.segDir, str(imgIndex+1).zfill(formats['segNLeadingZeros'])+'.'+self.formats['segFormat'])
		try:
			seg = np.asarray(Image.open(segPath))
			nInstances = np.max(seg)
			imgOverlap = img.copy()
			colorCycler = pltColorCycler()
			for iInstance in range(1,nInstances+1):
				colorString = next(colorCycler)
				color = np.array(mcolors.BASE_COLORS[colorString])*128
				imgOverlap[seg==iInstance] = img[seg==iInstance]//2 + color[None,None,:] #+ colorArr[iInstance-1,:][None,None,:]
			ax.imshow(imgOverlap)
		except FileNotFoundError:
			ax.imshow(img)

		# Scatter points and their covariances
		x = self.points[index]
		if x is not None:
			nInstances = x.shape[0]
			for iInstance in range(nInstances):
				ax.scatter(x[iInstance,:,0], x[iInstance,:,1])
				cov = self.covariance[index][iInstance]
				if cov is not None:
					for iKeypoint in range(cov.shape[0]):
						thisCov = cov[iKeypoint]
						(eigVal, eigVec) = np.linalg.eig(thisCov)
						for iDir in range(2):
							ax.arrow(x[iInstance,iKeypoint,0], x[iInstance,iKeypoint,1], np.sqrt(eigVal[iDir])*eigVec[0,iDir], np.sqrt(eigVal[iDir])*eigVec[1,iDir])

	def __len__(self):
		return self.len

	def read_img_np(self, imgPath):
		img = Image.open(imgPath).convert('RGB')
		img = np.array(img,np.uint8)
		return img


class PoseGenerator(AxesGenerator):
	def __init__(self, imgDir, formats, points, covariance, bbCorners, keypoints3D, K):
		# Input:
		# imgDir - directory containing N images
		self.imgDir = imgDir
		self.formats = formats
		self.points = points
		self.covariance = covariance
		self.bbCorners = bbCorners[0] # tuple: ([8,3] array, index list of plot order)
		self.cornerOrder = bbCorners[1]
		self.keypoints3D = keypoints3D
		self.K = K
		self.len = find_nbr_of_files(imgDir, self.formats['rgbFormat'])
		assert(self.len==len(points))
		assert(self.len==len(covariance))

	def __update_ax__(self, ax, index):
		print('asasdasdasdasd')
		index = index % self.len

		# Clear axes
		ax.clear()
		ax.set_title("Viewpoint {}".format(index))

		# Load image
		imgPath = os.path.join(self.imgDir, str(index+1).zfill(self.formats['rgbNLeadingZeros'])+'.'+self.formats['rgbFormat'])
		img = self.read_img_np(imgPath)
		height, width, _ = img.shape
		ax.imshow(img)

		# Scatter points and project poses
		x = self.points[index]
		if x is not None:
			nInstances = x.shape[0]
			colorCycler = pltColorCycler()
			for iInstance in range(nInstances):
				color = next(colorCycler)
				ax.scatter(x[iInstance,:,0], x[iInstance,:,1], c=color, edgecolors='black')
				covar = self.covariance[index][iInstance]
				if covar is not None:
					print(covar)
					weights = covar_to_weight(covar)
					pose = uncertainty_pnp(x[iInstance], weights, self.keypoints3D, self.K)
					poseBajs = pnp(self.keypoints3D, x[iInstance], self.K)
					print(pose)
					bbProj = project(self.bbCorners.T, self.K@pose)
					bbProjBajs = project(self.bbCorners.T, self.K@poseBajs)
					#ax.scatter(bbProj[:,0], bbProj[:,1], marker='*', c=color)
					ax.plot(bbProj[self.cornerOrder, 0], bbProj[self.cornerOrder, 1], c=color)
					ax.plot(bbProjBajs[self.cornerOrder, 0], bbProjBajs[self.cornerOrder, 1], c='brown')
					for iKeypoint in range(covar.shape[0]):
						thisCov = covar[iKeypoint]
						(eigVal, eigVec) = np.linalg.eig(thisCov)
						for iDir in range(2):
							ax.arrow(x[iInstance,iKeypoint,0], x[iInstance,iKeypoint,1], np.sqrt(eigVal[iDir])*eigVec[0,iDir], np.sqrt(eigVal[iDir])*eigVec[1,iDir])

		ax.set_xlim(0, width)
		ax.set_ylim(height, 0)



	def __len__(self):
		return self.len

	def read_img_np(self, imgPath):
		img = Image.open(imgPath).convert('RGB')
		img = np.array(img,np.uint8)
		return img



# SETTINGS
########################################################
resultsDirs = '/var/www/webdav/Data/ICA/Results/Final2'
sceneDirs = '/var/www/webdav/Data/ICA/Scenes/Train'
iScene = 1
iClass = 10
formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0, 'segFormat':'png', 'segNLeadingZeros':5}

# Implicit
resultsDir = os.path.join(resultsDirs, 'Scene'+str(iScene), 'Class'+str(iClass))
sceneDir = os.path.join(sceneDirs, 'Scene'+str(iScene), 'pvnet')
#sceneDir = '/var/www/webdav/Data/ICA/Scenes/Deprecated/SceneDeprecated1/pvnet'
paths = get_work_paths(sceneDir)
predSegDir = os.path.join(resultsDir, 'instanceMasks')
nViews = find_nbr_of_files(paths['rgbDir'], formats['rgbFormat'])



# EVALUTATION SCRIPT
##################################################

# Load 2D keypoints (Predicted)
points2DPath = os.path.join(resultsDir, 'points2D.pickle')
points2D = pickleLoad(points2DPath)
covariancePath = os.path.join(resultsDir, 'covariance.pickle')
covariance = pickleLoad(covariancePath)
# keepRatePath = os.path.join(resultsDir, 'settings', 'keepRate.pickle')
# keepRate = pickleLoad(keepRatePath)
keepRate = 4

# points2DPath = '/home/comvis/Temp/pvnetData/Send-Archive/points2D.pickle'
# points2D = pickleLoad(points2DPath)
# covariancePath = '/home/comvis/Temp/pvnetData/Send-Archive/covariance.pickle'
# covariance = pickleLoad(covariancePath)

# def load_pose_data(sceneDir, verbose=False):
# 	posesPath = os.path.join(sceneDir, 'poseannotation_out/poses.yml')
# 	poseData = yaml.load(open(posesPath, 'r'), Loader=yaml.UnsafeLoader)
# 	return poseData

# poseData = load_pose_data(sceneDir, verbose=True)
# instanceIdxPath = os.path.join(paths['segDir'], 'classindex.txt')
# instanceIdx = np.loadtxt(instanceIdxPath, delimiter=',')
# if instanceIdx.shape == (): # Special case for single instance
# 	instanceIdx.shape = (1,)
# idx = [i for i,j in enumerate(instanceIdx) if j==iClass] 
# poses = [] # List of length nViews where each element contains all gt poses 
# visibilityPath = os.path.join(sceneDir, 'visibility.yml')
# visibility = load_visibility_data(visibilityPath)
# for iView in range(nViews):
# 	thisPoses = [parse_pose(poseData, iView, iPose) for iPose in idx]
# 	poses.append(thisPoses)

# # Generate GT 2D points
# points2D_gt = []
# for iView in range(len())
	
# 	for iInstance in visibleInstanceIdx
# 		points2D_gt.append(thisPoints)

# Obtain the actual indices 
nViews = find_nbr_of_files(paths['rgbDir'], formats['rgbFormat'])
viewpointIdx = keep_every_nth(nViews, keepRate)

# Points on images
fig = plt.figure()
ax1 = fig.add_subplot(111)
rgbFormats = {'format':formats['rgbFormat'], 'nLeadingZeros':formats['rgbNLeadingZeros']}
axGen = ImageKeypointGenerator(paths['rgbDir'], rgbFormats, points2D, imgIdx=viewpointIdx)
plotScroller = PlotScroller(ax1, axGen)
cid = fig.canvas.mpl_connect('scroll_event', plotScroller.onscroll)
plt.show(block=False)

# # Instance masks on images
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# axGen = SegmentationOverlapGenerator(paths['rgbDir'], predSegDir, formats)
# plotScroller2 = PlotScroller(ax2, axGen)
# cid2 = fig2.canvas.mpl_connect('scroll_event', plotScroller2.onscroll)
# plt.show(block=False)

# Everything at once
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
axGen = SegKeypointsCovarianceGenerator(paths['rgbDir'], predSegDir, formats, points2D, covariance, imgIdx=viewpointIdx)
plotScroller3 = PlotScroller(ax3, axGen)
cid3 = fig3.canvas.mpl_connect('scroll_event', plotScroller3.onscroll)
plt.show()


# # # Compare vanilla pnp to uncertainty pnp for an image
# classNameToIdx, classIdxToName = create_class_idx_dict(paths['modelDir'])
# pathsClass = get_work_paths(sceneDir, classIdxToName[iClass], classNameToIdx)

# def corner_points(lowLims, highLims):
#     xLims = lowLims[0], highLims[0]
#     yLims = lowLims[1], highLims[1]
#     zLims = lowLims[2], highLims[2]
    
#     cornersList = [[xlim,ylim,zlim] for xlim in xLims for ylim in yLims for zlim in zLims]
#     corners = np.array(cornersList)
#     plotOrder = np.array([1,2,4,3,1,5,7,8,6,2,4,8,7,3,1,5,6])-1
#     return corners, plotOrder

# minBounds, maxBounds = get_model_limits(pathsClass['modelDir'], iClass)
# bbCorners = corner_points(minBounds, maxBounds) # Returns tuple
# K = parse_inner_parameters(paths['cameraPath'])
# keypoints3D = parse_3D_keypoints(pathsClass['keypointsPath'], addCenter=True) #[3, nKeypoints]
# keypoints3D = keypoints3D.T

# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111)
# axGen = PoseGenerator(pathsClass['rgbDir'], formats, points2D, covariance, bbCorners, keypoints3D, K)
# plotScroller4 = PlotScroller(ax4, axGen)
# cid4 = fig4.canvas.mpl_connect('scroll_event', plotScroller4.onscroll)
# plt.show()



