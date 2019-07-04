# Generate and save 2D GT keypoints

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
from lib.ica.ica_dataset import IcaDataset

# PVNet modules
from lib.utils.data_utils import read_rgb_np
from lib.utils.evaluation_utils import pnp
from lib.utils.extend_utils.extend_utils import uncertainty_pnp


# Settings
iScene = 13
className = 'tvalgron'
scenesDir = '/var/www/webdav/Data/ICA/Scenes/Validation'
formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0, 'segFormat':'png', 'segNLeadingZeros':5}

# Implicit
sceneDir = os.path.join(scenesDir, 'Scene'+str(iScene), 'pvnet')
modelDir = os.path.join(sceneDir, 'models')
classNameToIdx, classIdxToName = create_class_idx_dict(modelDir)
paths = get_work_paths(sceneDir, className, classNameToIdx)





classIdx = classNameToIdx[className]
keypointsPath = os.path.join(modelDir, str(classNameToIdx[className])+'_keypoints.txt')
keypoints = parse_3D_keypoints(keypointsPath, addCenter=True) # MAKE SURE THERE ARE KEYPOINTS.TXT FOR EVERY MODEL
nKeypoints = keypoints.shape[1]

# Create dummy scenesDir and symbolic link to the only sceneDir you want
tmpDir = '/tmp/scenesDir'
os.makedirs(tmpDir, exist_ok=True)
try:
	os.symlink(sceneDir, os.path.join(tmpDir, 'Scene'+str(iScene)))
except FileExistsError:
	pass

# Create the dataset
dataset = IcaDataset(classIdx, tmpDir, formats, keypoints, visibilityThreshold=0.3)
idxToSceneView = dataset.getIdxSceneViewMap()

nViews = find_nbr_of_files(paths['rgbDir'], formats['rgbFormat'])
points2D = [None]*nViews
poses = [None]*nViews
for index in range(dataset.len):
	_, iView = idxToSceneView[index]
	points, thisPoses = dataset.get_2d_points(index)
	# Filter out points that are out of bounds 
	points2D[iView-1] = points
	poses[iView-1] = thisPoses

pointsPath = '/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10/points2D_gt.pickle' 
posesPath = '/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10/poses_gt.pickle' 

pickleSave(points2D, pointsPath)
pickleSave(poses, posesPath)

