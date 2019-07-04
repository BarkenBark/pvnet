# Function to test the aggregation step given some 2D-keypoints

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
from lib.ica.run_utils import covar_to_weight, estimate_pose_center_ms

# PVNet modules
from lib.utils.data_utils import read_rgb_np
from lib.utils.evaluation_utils import pnp
from lib.utils.extend_utils.extend_utils import uncertainty_pnp


# SETTINGS
########################################################
resultsDirs = '/var/www/webdav/Data/ICA/Results/Experiment4'
sceneDirs = '/var/www/webdav/Data/ICA/Scenes/Validation'
iScene = 13
iClass = 10
formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0, 'segFormat':'png', 'segNLeadingZeros':5}

# Implicit
resultsDir = os.path.join(resultsDirs, 'Scene'+str(iScene), 'Class'+str(iClass))
sceneDir = os.path.join(sceneDirs, 'Scene'+str(iScene), 'pvnet')
#sceneDir = '/var/www/webdav/Data/ICA/Scenes/Deprecated/SceneDeprecated1/pvnet'
paths = get_work_paths(sceneDir)
classNameToIdx, classIdxToName = create_class_idx_dict(paths['modelDir'])
paths = get_work_paths(sceneDir, classIdxToName[iClass], classNameToIdx)
segDir = os.path.join(resultsDir, 'instanceMasks')

# EVALUTATION SCRIPT
##################################################

# Plot 2D keypoints
points2DPath = os.path.join(resultsDir, 'points2D.pickle')
#points2DPath = '/home/comvis/Temp/pvnetData/Send-Archive/points2D.pickle'
points2D = pickleLoad(points2DPath)
covariancePath = os.path.join(resultsDir, 'covariance.pickle')
#covariancePath = '/home/comvis/Temp/pvnetData/Send-Archive/covariance.pickle'
covariance = pickleLoad(covariancePath)
#covariance = None
points3D = parse_3D_keypoints(paths['keypointsPath'], addCenter=True) #[3, nKeypoints]
points3D = points3D.T
K = parse_inner_parameters(paths['cameraPath'])
motion = parse_motion(paths['motionPath'])
aggregationSettings = {'inlierThreshold':7, 'bandwidthSweepResolution':50, 'binFreqMultiplier':1/50}
aggregationSettings['cameraMatrix'] = K
aggregationSettings['classIdx'] = iClass

logDir = None
#logDir = '/var/www/webdav/Data/ICA/Results/Experiment5' + '/Scene14/Class11'
logDir = '/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10'
print(len(points2D))
estimate_pose_center_ms(points2D, points3D, covariance, motion, paths, aggregationSettings, plotCenters=True, logDir=logDir)

