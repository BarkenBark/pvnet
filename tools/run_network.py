# Script to run network on a scene to produce segmentations and vertex fields

# Import statements
#######################################
import os
import sys
import time
import pickle
import argparse
import time
sys.path.append('.')
sys.path.append('..')

# Own modules
from lib.ica.utils import *
from lib.ica.run_utils import load_network, run_network
#from tools.train_ica_full import IcaDataset

# Other modules
import torch
from torch import cuda
from PIL import Image
from matplotlib import pyplot as plt




# PARSE ARGUMENTS
#############################################
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--targetDir', help='Target directory to store output images')
parser.add_argument('-i', '--sceneIdx', help='Index of scene')
args = parser.parse_args()
targetDir = args.targetDir
sceneIdx = args.sceneIdx


# SETTINGS
######################################################
 
scenesDir = '/var/www/webdav/Data/ICA/Scenes/Train' # Directory containing scene directories labeled SceneX, X=1,2,...
iKeypoint = 0 # Keypoint index for which to visualize vertex fields
classIdx = 11 # If you only want results for one class, have this not be None
rgbFormats = {'rgbFormat': 'png', 'rgbNLeadingZeros': 0}





# MAIN SCRIPT
####################################

sceneDir = os.path.join(scenesDir, 'Scene'+str(sceneIdx), 'pvnet')
paths = get_work_paths(sceneDir)
_, classIdxToName = create_class_idx_dict(paths['modelDir'])
nClasses = len(classIdxToName)
nImages = find_nbr_of_files(paths['rgbDir'], format=rgbFormats['rgbFormat'])
try:
	os.mkdir(os.path.join(targetDir, 'seg'))
	os.mkdir(os.path.join(targetDir, 'ver'))
except FileExistsError:
	pass
	


# for iClass in classes:

# 	# Add class network and keypoints paths to paths
# 	className = classIdxToName[iClass]
# 	paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
# 	paths['keypointsPath'] = os.path.join(paths['modelDir'], str(iClass)+'_keypoints.txt')

# 	# Load network
# 	network = load_network(paths)

# 	# Run the network
# 	classMask = []
# 	classAngles = []
# 	for iImage in range(1, nImages+1):
# 		segPred, verPred = run_network(network, paths, rgbFormats, rgbIdx=iImage)
# 		thisClassMask = segpred_to_mask(segPred)
# 		classMasks.append(thisClassMask)
# 		thisClassAngles = vertexfield_to_angles(verPred, thisClassMask, keypointIdx=iKeypoint)
# 		classAngles.append(thisClassAngles)

if classIdx is not None:
	rangeObj = range(classIdx, classIdx+1)
else:
	rangeObj = range(1, nClasses+1)

# Pre-load the networks
networks = {}
for iClass in rangeObj:

	# Add class network and keypoints paths to paths
	className = classIdxToName[iClass]
	paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
	paths['keypointsPath'] = os.path.join(paths['modelDir'], str(iClass)+'_keypoints.txt')

	# Check if a network exists for the class. If not, don't do shit
	if not os.path.isfile(paths['networkPath']):
		continue

	network = load_network(paths)
	networks[iClass]=network

for iImage in range(1, nImages+1):

	# Calculate network outputs for all classes in the image
	classMask = {}
	classAngles = {}
	for iClass in rangeObj:

		# # Add class network and keypoints paths to paths
		# className = classIdxToName[iClass]
		# paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
		# paths['keypointsPath'] = os.path.join(paths['modelDir'], str(iClass)+'_keypoints.txt')

		# # Check if a network exists for the class. If not, don't do shit
		# if not os.path.isfile(paths['networkPath']):
		# 	continue

		# Load network
		network = networks[iClass]

		# Run the network

		t = time.time()
		segPred, verPred = run_network(network, paths, rgbFormats, rgbIdx=iImage)
		print('run_network time elapsed: {}'.format(time.time()-t))
		segPred = segPred.cpu().detach().numpy()
		verPred = verPred.cpu().detach().numpy()

		t2 = time.time()
		thisClassMask = segpred_to_mask(segPred)
		classMask[iClass] = thisClassMask
		thisClassAngles = vertexfield_to_angles(verPred, thisClassMask, keypointIdx=iKeypoint)
		classAngles[iClass] = thisClassAngles
		print('Other shit time elapsed: {}'.format(time.time()-t2))


	# Create the seg and ver image
	print('......................')
	t3 = time.time()
	if classIdx is not None:
		segImg = Image.fromarray(classMask[iClass][0,:,:])
		segTargetPath = os.path.join(targetDir, 'seg', str(iImage).zfill(5)+'.png')
		segImg.save(segTargetPath)
		plt.imshow(classAngles[iClass])
		plt.hsv()
		verTargetPath = os.path.join(targetDir, 'ver', str(iImage).zfill(5)+'.png')
		plt.savefig(verTargetPath)
		plt.clf()

	print('Saving image time elapsed: {}'.format(time.time()-t3))




	print('Fininshed image {}/{}.'.format(iImage, nImages))



