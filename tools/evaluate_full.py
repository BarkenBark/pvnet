# Script for evaluating the results across all scenes

# Import statements
#######################################
import os
import sys
import time
import pickle
sys.path.append('.')
sys.path.append('..')

# Own modules
from lib.ica.utils import *
from lib.ica.run_utils import load_network, calculate_points_multiple_views, estimate_pose, estimate_pose_center_ms

# Other modules
from torch import cuda

# Configure warnings handling
import warnings
warnings.filterwarnings("ignore", message="nn.UpsamplingBilinear2d is deprecated.")


# SETTINGS
########################################################################

# Path settings
scenesDir = '/var/www/webdav/Data/ICA/Scenes/Train' # Scenes on which to run the pipeline. Contains arbitrarily named Scene directories, and nothing else.
resultsDir = 'var/www/webdav/ICA/Results/Final' # Contains folders with structure SceneX/ClassY
formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0}

# Implicit settings
nScenes = find_nbr_scenes(scenesDir)





# Create one list of absolute paths to specific scene directories. --- WARNING: DO NOT USE os.listdir() ON scenesDir AGAIN, AS IT MAY RETURN DIRECTORIES IN A RANDOM ORDER
sceneDirs = [os.path.join(scenesDir, sceneDirName, 'pvnet') for sceneDirName in sorted(os.listdir(scenesDir))]


poses = [] # List of lists of lists of poses. poses[i][j][k] is the pose of the kth instance of the jth class in the ith scene
results = []
for iScene in range(3,4):
	thisSceneDir = sceneDirs[iScene]
	print('Starting the pipeline for Scene {}/{}'.format(iScene+1, nScenes))
	scenePoses = pipeline(thisSceneDir)
	poses.append(scenePoses)
	if evaluate:
		posesGT = load_gt(thisSceneDir) # TO-DO: Implement load_gt()
		results.append(evaluate(poses, posesGT))

# Plot the results
with open('/var/www/webdav/Data/ICA/Results/poses.pickle', 'wb') as file:
	pickle.dump(poses, file)