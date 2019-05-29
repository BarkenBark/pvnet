# Main script for running the entire pipeline on each scene, and evaluating the results across all scenes

# Import statements
#######################################
import os
import sys
import time
sys.path.append('.')
sys.path.append('..')

# Own modules
from lib.ica.utils import *
from lib.ica.run_utils import load_network, calculate_points_multiple_views, estimate_pose

# Other modules
from torch import cuda





# SETTINGS
########################################################################

# Choose whether to evaluate against ground truth or not
evaluate = False

# Path settings
testSceneDir = '/var/www/webdav/Data/ICA/Scenes/Train' # Contains arbitrarily named Scene directories, and nothing else
formats = {'rgbFormat':'jpg', 'rgbNLeadingZeros':0}

# Method settings
ransacSettings = {'nHypotheses':1024, 'threshold':0.999} # For RANSAC on pvnet output
nmsSettings = {'similariyThreshold':1/15, 'scoreThreshold': 180 , 'neighborThreshold':40} # For NMS on 2D center point. # TO-DO: Calculate actual values used in NMS by looking at max(linlierCounts)
instanceSegSettings = {'thresholdMultiplier':0.9, 'discardThresholdMultiplier':0.7} # For instance segmentation based on center point predictions.
aggregationSettings = {'inlierThreshold':7}
keepRate = 10 # Only compute every keepRate:th viewpoint for 2D point calculations. # TO-DO: Allow for different skiprates for different scenes

# Implicit settings
nScenes = find_nbr_scenes(testSceneDir)







# Function: Run the pipeline to get poses in the scene
##########################################################

def pipeline(sceneDir):

	# Load common values for this scene
	paths = get_work_paths(sceneDir)
	_, classIdxToName = create_class_idx_dict(paths['modelDir'])
	K = parse_inner_parameters(paths['cameraPath'])
	aggregationSettings['cameraMatrix'] = K

	# Motion
	motionFull = parse_motion(os.path.join(paths['poseOutDir'], 'motionScaled.txt'))
	nCamerasFull = find_nbr_of_files(paths['rgbDir'], formats['rgbFormat'])
	assert(len(motionFull) == nCamerasFull)
	if keepRate > 0:
		viewpointIdx = [idx*keepRate for idx in range(nCamerasFull) if idx < nCamerasFull/keepRate]
		motion = [motionFull[idx] for idx in viewpointIdx]
		nCameras = len(motion)
	else:
		motion = motionFull
	
	# TO-DO: Remove
	classIdxToName = {1:'tval', 2:'seltin'}

	# Find instances and their poses for each class in this scene
	#for iClass in range(1, nClasses+1):
	poses = []
	for iClass in range(1, 3):

		# Add class network and keypoints paths to paths
		className = classIdxToName[iClass]
		paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
		paths['keypointsPath'] = os.path.join(paths['poseOutDir'], 'keypoints', className+'_keypoints.txt')

		# Load the 3D keypoints
		points3D = parse_3D_keypoints(paths['keypointsPath'], addCenter=True)

		# Load the network
		network = load_network(paths) # FIX: network needs to be loaded separately for each class

		# Predict the 2D keypoints in each viewpoint
		points2D, covariance = calculate_points_multiple_views(viewpointIdx, network, paths, formats, ransacSettings, nmsSettings, instanceSegSettings)
		cuda.empty_cache() 

		# Finally, calculate poses
		classPoses = estimate_pose(points2D, points3D, motion, aggregationSettings) # List of poses of length nInstances
		poses.append(classPoses)

	return poses # List of lists of poses. poses[i][j] is the pose of the jth instance of the ith class 









# MAIN SCRIPT
########################################################

poses = [] # List of lists of lists of poses. poses[i][j][k] is the pose of the kth instance of the jth class in the ith scene
results = []
for iScene in range(1,nScenes+1):
	thisSceneDir = os.path.join(testSceneDir, 'Scene'+str(iScene), 'pvnet')
	print('Starting the pipeline for Scene {}'.format(iScene))
	scenePoses = pipeline(thisSceneDir)
	poses.append(scenePoses)
	if evaluate:
		posesGT = load_gt(thisSceneDir) # TO-DO: Implement load_gt()
		results[iScene] = evaluate(poses, posesGT)

# Plot the results

