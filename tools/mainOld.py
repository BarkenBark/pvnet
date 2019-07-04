ma# Main script for running the entire pipeline on each scene, and evaluating the results across all scenes

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
from lib.ica.run_utils import load_network, calculate_points_multiple_views, estimate_pose, estimate_pose_center_ms, get_gaussian_distance_similarity, inner_product_exponentiated

# Other modules
from torch import cuda

# Configure warnings handling
import warnings
warnings.filterwarnings("ignore", message="nn.UpsamplingBilinear2d is deprecated.")


# SETTINGS
########################################################################

# Choose wether to resume evaluation or start from scratch
resume = False

# Choose whether to evaluate against ground truth or not
evaluate = False

# Enable/disable saving intermediate results (NONE means no logging)
logDir = None
logDir = '/var/www/webdav/Data/ICA/Results/July3'

# Path settings
scenesDir = '/var/www/webdav/Data/ICA/Scenes/Train' # Scenes on which to run the pipeline. Contains arbitrarily named Scene directories, and nothing else.
formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0}

# Method settings
ransacSettings = {'nHypotheses':1024, 'threshold':0.999} # For RANSAC on pvnet output
keypointPixelRadius = 20
voting_function = lambda x,y: get_gaussian_distance_similarity(x,y,distance01=keypointPixelRadius) * inner_product_exponentiated(x, y, innerProductExponent = 1, frequencyMultiplierExponent = 0, threshold = None)
detectionSettings = {'keypointPixelRadius':keypointPixelRadius, 'minBinMultiplier':1/50, 'clusterMeanMaxRadius':4}
#nmsSettings = {'similariyThreshold':1/15, 'scoreThreshold': 180 , 'neighborThreshold':40} # For NMS on 2D center point. # TO-DO: Calculate actual values used in NMS by looking at max(linlierCounts)
#instanceSegSettings = {'thresholdMultiplier':0.9, 'discardThresholdMultiplier':0.7} # For instance segmentation based on center point predictions.
#aggregationSettings = {'inlierThreshold':7, 'bandwidthSweepResolution':50, 'msMinBinFrequency':3, 'msMaxBinFrequency':7}
aggregationSettings = {'inlierThreshold':7, 'bandwidthSweepResolution':50, 'binFreqMultiplier':1/50}
keepRate = 4 # Only compute every keepRate:th viewpoint for 2D point calculations. # TO-DO: Allow for different skiprates for different scenes

# Store the settings
if logDir is not None:
	settingsLogDir = os.path.join(logDir, 'settings')
	pickleSave(ransacSettings, os.path.join(settingsLogDir, 'ransacSettings.pickle'))
	pickleSave(detectionSettings, os.path.join(settingsLogDir, 'detectionSettings.pickle'))
	pickleSave(aggregationSettings, os.path.join(settingsLogDir, 'aggregationSettings.pickle'))
	pickleSave(keepRate, os.path.join(settingsLogDir, 'keepRate.pickle'))

detectionSettings['voting_function'] = voting_function # Do this after storing settings because cant save with pickle otherwise

# Implicit settings
nScenes = find_nbr_scenes(scenesDir)








# Function: Run the pipeline to get poses in the scene
##########################################################

def pipeline(sceneDir):

	# Load common values for this scene
	paths = get_work_paths(sceneDir)
	sceneName = get_scene_name(sceneDir)
	classNameToIdx, classIdxToName = create_class_idx_dict(paths['modelDir'])
	nClasses = len(classIdxToName)
	K = parse_inner_parameters(paths['cameraPath'])
	aggregationSettings['cameraMatrix'] = K

	# Load (and downsample) motion
	motionFull = parse_motion(os.path.join(paths['poseOutDir'], 'motionScaled.txt'))
	nCamerasFull = find_nbr_of_files(paths['rgbDir'], formats['rgbFormat'])
	assert(len(motionFull) == nCamerasFull)
	viewpointIdx = keep_every_nth(nCamerasFull, keepRate)
	# if keepRate > 0:
	# 	viewpointIdx = [idx*keepRate+1 for idx in range(nCamerasFull) if idx < nCamerasFull/keepRate]
	# else:
	# 	viewpointIdx = [idx+1 for idx in range(nCamerasFull)]
	motion = [motionFull[idx-1] for idx in viewpointIdx]
	nCameras = len(motion)

	# Find instances and their poses for each class in this scene
	poses = []
	for iClass in range(1,nClasses+1):

		# Add class network and keypoints paths to paths
		className = classIdxToName[iClass]
		print('Running for class '+className)
		tStart = time.time()
		# paths['networkPath'] = os.path.join(paths['networkDir'], className, className+'Network.pth')
		# paths['keypointsPath'] = os.path.join(paths['modelDir'], str(iClass)+'_keypoints.txt')
		paths = get_work_paths(sceneDir, className, classNameToIdx)

		# Check if a network exists for the class. If not, don't predict any poses.
		if not os.path.isfile(paths['networkPath']):
			poses.append(None)
			#print('No network found for class '+ className + '. Skipping this.')
			continue

		# Create log directory for this scene/class
		if logDir is not None:
			thisLogDir = os.path.join(logDir, sceneName, 'Class'+str(iClass))
			try: 
				os.makedirs(thisLogDir)
			except FileExistsError:
				pass
		else:
			thisLogDir = None

		# Load the 3D keypoints
		points3D = parse_3D_keypoints(paths['keypointsPath'], addCenter=True) #[3, nKeypoints]
		points3D = points3D.T

		# Load the network
		network = load_network(paths) # FIX: network needs to be loaded separately for each class

		# Predict the 2D keypoints in each viewpoint
		points2D, covariance = calculate_points_multiple_views(viewpointIdx, network, paths, formats, ransacSettings, detectionSettings, plotView=False, logDir=thisLogDir, verbose=True)
		cuda.empty_cache() 
		print("Finished class "+className+" after {} seconds".format(time.time()-tStart))

		# Finally, calculate poses
		aggregationSettings['classIdx'] = iClass
		if thisLogDir is not None:
			pickleSave((points2D, points3D, covariance, motion, paths, aggregationSettings), os.path.join(thisLogDir, 'state'))
		classPoses = estimate_pose_center_ms(points2D, points3D, covariance, motion, paths, aggregationSettings, plotCenters=False, logDir=thisLogDir) # List of poses of length nInstances
		nClassInstances = len(classPoses)
		print('Detected {} instances of class {}.'.format(nClassInstances, className))
		#input('Press enter to continue')
		poses.append(classPoses)
		if thisLogDir is not None:
			pickleSave(classPoses, os.path.join(thisLogDir, 'finalClassPoses.pickle'))

	if logDir is not None:
		pickleSave(poses, os.path.join(logDir, sceneName, 'finalPoses.pickle'))
	return poses # List of lists of poses. poses[i][j] is the pose of the jth instance of the ith class 









# MAIN SCRIPT
########################################################

# Create one list of absolute paths to specific scene directories. --- WARNING: DO NOT USE os.listdir() ON scenesDir AGAIN, AS IT MAY RETURN DIRECTORIES IN A RANDOM ORDER
sceneDirs = [os.path.join(scenesDir, sceneDirName, 'pvnet') for sceneDirName in sorted(os.listdir(scenesDir))]

poses = [] # List of lists of lists of poses. poses[i][j][k] is the pose of the kth instance of the jth class in the ith scene
results = []
for iScene in range(nScenes):
	thisSceneDir = sceneDirs[iScene]
	sceneName = get_scene_name(thisSceneDir)
	print('Starting the pipeline for Scene {}/{} ({})'.format(iScene+1, nScenes, sceneName))
	scenePoses = pipeline(thisSceneDir)
	poses.append(scenePoses)
	if evaluate:
		posesGT = load_gt(thisSceneDir) # TO-DO: Implement load_gt()
		results.append(evaluate(poses, posesGT))

# Save the results
if logDir is not None:
	pickleSave(poses, os.path.join(logDir, 'posesFinalFinal.pickle'))
