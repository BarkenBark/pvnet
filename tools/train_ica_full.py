# IMPORT STATEMENTS
#################################################
import os
import sys
import time
import argparse
sys.path.append('.')
sys.path.append('..')

# Import own modules
from lib.ica.utils import * # Includes parse_3D_keypoints, pflat, pextend etc. 
from lib.ica.ica_dataset import IcaDataset

# Import pvnet modules
from lib.networks.model_repository import Resnet18_8s
from lib.utils.data_utils import read_rgb_np
from lib.utils.net_utils import smooth_l1_loss, compute_precision_recall
from lib.utils.draw_utils import visualize_mask, visualize_vertex_field
from lib.datasets.linemod_dataset import compute_vertex_hcoords

# Import other modules
import torch
from torch.utils.data import Dataset, RandomSampler, BatchSampler, DataLoader
from torch.nn import DataParallel, CrossEntropyLoss
from torch.optim import Adam
import torchvision.transforms as transforms # Used for converting input rgb to tensor, and normalizing to values suitable for ImageNet-trained models
import numpy as np
from numpy import zeros
from PIL import Image
from numpy import eye, unique
import yaml

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd



# PARSE ARGUMENTS
###############################################

# Parse classname
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classname', help='Name of the class for which to train a network')
parser.add_argument('--resumeTraining',  action='store_true')
args = parser.parse_args()
className = args.classname
if className is None:
	print('WARNING: No class specified. Using class specified in file.')
	print('Exiting program.')
	exit()
















# SETTINGS
#################################

# Paths
dataDir = '/var/www/webdav/Data/ICA/'
# if className is None:
# 	className = 'makaroner' # MAYBE REPLACE WITH CLASSIDX NUMBER?
formats = {'rgbFormat':'png', 'rgbNLeadingZeros':0, 'segFormat':'png', 'segNLeadingZeros':5}
#checkpointPath = '/var/www/webdav/Data/ICA/Results/results.pickle' # For loss series

# Misc
# skipInvisible = True # True => Images in which instances of the class cannot be seen due to truncation (out of bounds) are not included in the training
visThreshold = 0.3 # The minimum visibility value for an instance to be considered visibile in an image

# Training parameters
nEpochs = 1000
resumeTraining = args.resumeTraining
batchSize = 16
lossRatio = 0.5 # Weight ratio of vertex loss and segmentation loss (1 is all vertex, 0 is all segmentation)
patience = 5
minDeltaFactor = 0.05

# Optimization parameters
learningRate = 1e-3



# Implicit variables (TODO: Parse from argument or settings file)
#################################

# Implicit paths 
trainScenesDir = os.path.join(dataDir, 'Scenes', 'Train')
valScenesDir = os.path.join(dataDir, 'Scenes', 'Validation')
modelDir = os.path.join(dataDir, 'Models', 'models_meter')
classNameToIdx, classIdxToName = create_class_idx_dict(modelDir)
try: # Assert that the specified className is valid before proceeding
	classNameToIdx[className]
except KeyError:
	print('The specified class name *'+className+'* is not valid. (Can\'t find in model directory)')
	print('Exiting program.')
	exit()

# If networkDir does not exist, create it
try:
	networkDir = os.path.join(dataDir, 'Networks', className)
	os.mkdir(networkDir)
except FileExistsError:
	print('Network directory already exists.')
	if not resumeTraining:
		print('##########################################################################')
		print('# WARNING: Training from scratch will overwrite the current network and checkpoint in ' + networkDir)
		print('##########################################################################')
		# proceed = input('Are you sure that you want to proceed? (y/n): ')
		# if proceed != 'y':
		# 	print('Exiting program.')
		# 	exit()

checkpointPath = os.path.join(networkDir, 'checkpoint.pth')
bestNetPath = os.path.join(networkDir, className+'Network.pth')

# Implicit values
keypointsPath = os.path.join(modelDir, str(classNameToIdx[className])+'_keypoints.txt')
keypoints = parse_3D_keypoints(keypointsPath, addCenter=True) # MAKE SURE THERE ARE KEYPOINTS.TXT FOR EVERY MODEL
nKeypoints = keypoints.shape[1]

















# # DATASET CLASS DEFINITION
# ###############################

# # Helper functions
# def load_inner_parameters(sceneDirs):
# 	K = []
# 	for sceneDir in sceneDirs:
# 		cameraPath = os.path.join(sceneDir, 'rgb/camera.yml')
# 		thisK = parse_inner_parameters(cameraPath)
# 		K.append(thisK)
# 	return K
# def load_pose_data(sceneDirs, verbose=False):
# 	poseData = []
# 	nScenes = len(sceneDirs)
# 	for iScene, sceneDir in enumerate(sceneDirs):
# 		if verbose:
# 			sys.stdout.write("\rReading pose data for scene {}/{}".format(iScene+1,nScenes))
# 			sys.stdout.flush()
# 		posesPath = os.path.join(sceneDir, 'poseannotation_out/poses.yml')
# 		thisPoseData = yaml.load(open(posesPath, 'r'), Loader=yaml.UnsafeLoader)
# 		poseData.append(thisPoseData)
# 	if verbose: sys.stdout.write('\n')
# 	return poseData
# def load_visibility_data(visibilityPath):
# 	with open(visibilityPath, 'r') as f:
# 		visibilityData = yaml.load(f, Loader=yaml.UnsafeLoader)
# 	return visibilityData

# # ICA Dataset Class (Complete with init, getitem and len)
# class IcaDataset(Dataset):
# 	def __init__(self, classIdx, scenesDir, formats, keypoints, skipInvisible=False):

# 		# Trivial class member initilizations
# 		self.classIdx = classIdx # NOTE: Indexed from 1 to nClasses
# 		self.formats = formats # TO-DO: Deprecate
# 		self.keypoints = keypoints

# 		# Create one list of absolute paths to specific scene directories. --- WARNING: DO NOT USE os.listdir() ON scenesDir AGAIN, AS IT MAY RETURN DIRECTORIES IN A RANDOM ORDER
# 		self.sceneDirs = [os.path.join(scenesDir, sceneDirName, 'pvnet') for sceneDirName in sorted(os.listdir(scenesDir))]

# 		# Load inner parameters (just choose the K matrix of the first scene)
# 		self.K = load_inner_parameters(self.sceneDirs)

# 		# Load poseData for each scene (list of length nScenes)
# 		self.poseData = load_pose_data(self.sceneDirs, verbose=True)

# 		# Define transformation to normalize input images with ImageNet values
# 		self.test_img_transforms = transforms.Compose([
# 			transforms.ToTensor(), # if image.dtype is np.uint8, then it will be divided by 255
# 			transforms.Normalize(mean=imageNetMean,
# 								 std=imageNetStd)
# 		])

# 		# Store formats for each individual scene in a list of length nScenes
# 		self.formatList = parse_rgb_formats(self.sceneDirs)

# 		# Check number of images across all Scene directories to determine self.len
# 		# Also build self.idxToSceneView, mapping sampler index to scene and viewpoint index
# 		self.len = 0
# 		self.idxToSceneView = []
# 		nScenes = len(self.sceneDirs)
# 		for iScene in range(nScenes):
# 			sceneDir = self.sceneDirs[iScene]
# 			thisRgbDir = os.path.join(sceneDir, 'rgb')
# 			thisSegDir = os.path.join(sceneDir, 'seg')
# 			thisNFiles = find_nbr_of_files(thisRgbDir, format=self.formatList[iScene])
# 			assert(thisNFiles == find_nbr_of_files(thisSegDir, format=self.formats['segFormat']))
# 			if skipInvisible:
# 				visibilityPath = os.path.join(sceneDir, 'visibility.yml')
# 				visibility = load_visibility_data(visibilityPath)
# 				instanceIdx = [iInstance for iInstance, instanceDict in enumerate(self.poseData[iScene][0]) if instanceDict['obj_id']==self.classIdx] # Assuming the same instances have entries for every viewpoint
# 				if instanceIdx == []:
# 					classVisibility = [0]*thisNFiles
# 				else:
# 					classVisibility = [max([visibility[iImg][iInstance]['visibility'] for iInstance in instanceIdx]) for iImg in range(thisNFiles)] # List over best visibilities in all images in the scene
# 				thisNVisibleFiles = sum(1 for vis in classVisibility if vis >= visThreshold)
# 				self.len += thisNVisibleFiles
# 				self.idxToSceneView += [(iScene+1, iImg+1) for iImg in range(thisNFiles) if classVisibility[iImg] >= visThreshold]
# 				print('Found ' + str(thisNVisibleFiles) + ' images in ' + thisRgbDir + ' where at least one instance is visible.')
# 			else:
# 				self.len += thisNFiles
# 				self.idxToSceneView += [(iScene+1,iImg+1) for iImg in range(thisNFiles)]
# 				print('Found ' + str(thisNFiles) + ' images in ' + thisRgbDir)

# 		# Assert conditions
# 		assert(self.len == len(self.idxToSceneView))

# 	def __getitem__(self, index):
# 		# OUTPUTS:
# 		# rgb - input rgb tensor [3, height, width]
# 		# mask - ground truth segmentation tensor [2, height, width]
# 		# ver - ground truth vertex tensor [2*nKeypoints, height, width]
# 		# verWeights - ground truth binary weight tensor [_, height, width]

# 		# Decide which scene and which image this sample index corresponds to
# 		sceneIdx, viewIdx = self.idxToSceneView[index]
# 		sceneDir = self.sceneDirs[sceneIdx-1]

# 		# RGB
# 		rgbPath = os.path.join(sceneDir, 'rgb', str(viewIdx).zfill(self.formats['rgbNLeadingZeros'])+'.'+self.formatList[sceneIdx-1])
# 		rgb = read_rgb_np(rgbPath)
# 		rgb = self.test_img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))
# 		_, height, width = rgb.shape

# 		# Mask
# 		segDir = os.path.join(sceneDir, 'seg')
# 		segPath = os.path.join(segDir, str(viewIdx).zfill(self.formats['segNLeadingZeros'])+'.'+self.formats['segFormat'])
# 		segImg = np.asarray(Image.open(segPath))
# 		instanceIdxPath = os.path.join(segDir, 'classindex.txt')
# 		instanceIdx = np.loadtxt(instanceIdxPath, delimiter=',')
# 		if instanceIdx.shape == (): # Special case for single instance
# 			instanceIdx.shape = (1,)

# 		idxMatch = (instanceIdx == self.classIdx).astype(int)
# 		idxMatch = np.concatenate(([0], idxMatch)) # Add 'no-class' idx
# 		mask = torch.tensor(idxMatch[segImg], dtype=torch.int64)

# 		# Vertex
# 		nKeypoints = (self.keypoints).shape[1]
# 		ver = np.zeros([height,width,nKeypoints*2],np.float32)
# 		instanceSegImg = mask.numpy()*segImg
# 		nInstances = sum(idxMatch)

# 		idx = [i for i,j in enumerate(instanceIdx) if j==self.classIdx] 
# 		poses = [parse_pose(self.poseData[sceneIdx-1], viewIdx-1, iPose) for iPose in idx]

# 		for iInstance in range(nInstances):
# 			thisMask = instanceSegImg == idx[iInstance]+1
# 			keypointsProjected = self.K[sceneIdx-1] @ (poses[iInstance] @ pextend(self.keypoints, vecType='col'))
# 			keypointsProjected = pflat(keypointsProjected)
# 			ver = ver + compute_vertex_hcoords(thisMask, keypointsProjected.T)

# 		ver=torch.tensor(ver, dtype=torch.float32).permute(2, 0, 1)
# 		verWeights=mask.unsqueeze(0).float()

# 		return rgb, mask, ver, verWeights

# 	def __len__(self):
# 		return self.len

















# TRAINING AND VALIDATION FUNCTION DEFINITIONS
################################################################################################

def train(network, trainLoader, optimizer, iterPause=0):

	network.train()
	lossSegTotal = 0
	lossVertexTotal = 0
	lossTotal = 0

	nIterations = len(trainLoader.dataset) // trainLoader.batch_sampler.batch_size
	tTrainingLoopStart = time.time()
	for idx, data in enumerate(trainLoader):
		
		# Extract data
		tExtractDataStart = time.time()
		image, maskGT, vertexGT, vertexWeightsGT = [d.cuda() for d in data]
		#image = data[0].cuda(device = torch.device('cuda:0'))
		#maskGT = data[1].cuda(device = torch.device('cuda:0'))
		#vertexGT = data[2].cuda(device = torch.device('cuda:1'))
		#vertexWeightsGT = data[3].cuda(device = torch.device('cuda:1'))

		tExtractDataElapsed = time.time() - tExtractDataStart

		# Forward propagate
		tForwardPropStart = time.time()
		segPred, vertexPred = network(image)
		tForwardPropElapsed = time.time() - tForwardPropStart

		# Compute loss
		tComputeLossStart = time.time()
		criterion = CrossEntropyLoss(reduce=False) # Imported from torch.nn
		lossSeg = criterion(segPred, maskGT)
		lossSeg = torch.mean(lossSeg.view(lossSeg.shape[0],-1),1)
		lossVertex = smooth_l1_loss(vertexPred, vertexGT, vertexWeightsGT, reduce=False)
		#precision, recall = compute_precision_recall(segPred, maskGT)
		lossSeg = torch.mean(lossSeg) # Mean over batch
		lossVertex = torch.mean(lossVertex) # Mean over batch
		loss = (1-lossRatio)*lossSeg + lossRatio*lossVertex
		tComputeLossElapsed = time.time() - tComputeLossStart

		# # test custom score
		# nBatches = len(maskGT)
		# #[print_attributes('shape', variables=x) for x in (maskGT, vertexGT, vertexWeightsGT, segPred, vertexPred)]
		
		# for iBatch in range(nBatches):
		# 	valScores.append(custom_net_score(maskGT[iBatch:iBatch+1], vertexGT[iBatch:iBatch+1], vertexWeightsGT[iBatch:iBatch+1], segPred[iBatch:iBatch+1], vertexPred[iBatch]))

		# Update weights
		tUpdateWeightStart = time.time()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		tUpdateWeightElapsed = time.time() - tUpdateWeightStart

		# Compute moving average losses
		lossSegTotal = (lossSegTotal*idx + lossSeg.item())/(idx+1)
		lossVertexTotal = (lossVertexTotal*idx + lossVertex.item())/(idx+1)
		lossTotal = (lossTotal*idx + loss.item())/(idx+1)

		# Print training loop iteration time(s)
		tTrainingLoopElapsed = time.time() - tTrainingLoopStart
		if (idx % int(nIterations/10))==0:
			# print('Extracting data took ' + str(tExtractDataElapsed) + ' seconds.')
			# print('Forward propagating took ' + str(tForwardPropElapsed) + ' seconds.')
			# print('Computing loss took ' + str(tComputeLossElapsed) + ' seconds.')
			# print('Updating weights took ' + str(tUpdateWeightElapsed) + ' seconds.')
			# print('Individual steps took at total of {} seconds.'.format(tExtractDataElapsed+tForwardPropElapsed+tComputeLossElapsed+tUpdateWeightElapsed))
			print('Ended training loop iteration {}/{}, elapsed time is {} seconds.'.format(idx, nIterations ,tTrainingLoopElapsed))
			if idx != 0:
				print('Expected time until end of training epoch: {} seconds'.format((nIterations/idx-1)*tTrainingLoopElapsed + iterPause))

	return lossTotal, lossVertexTotal, lossSegTotal

def validate(network, valLoader):

	lossSegTotal = 0
	lossVertexTotal = 0
	lossTotal = 0
	network.eval()

	#nIterations = len(valLoader.dataset) // valLoader.batch_sampler.batch_size
	for idx, data in enumerate(valLoader):

		with torch.no_grad():

			# Extract data and forward propagate
			image, maskGT, vertexGT, vertexWeightsGT = [d.cuda() for d in data]
			segPred, vertexPred = network(image)

			# Compute loss
			criterion = CrossEntropyLoss(reduce=False) # Imported from torch.nn
			lossSeg = criterion(segPred, maskGT)
			lossSeg = torch.mean(lossSeg.view(lossSeg.shape[0],-1),1)
			lossVertex = smooth_l1_loss(vertexPred, vertexGT, vertexWeightsGT, reduce=False)
			precision, recall = compute_precision_recall(segPred, maskGT)
			lossSeg = torch.mean(lossSeg) # Mean over batch
			lossVertex = torch.mean(lossVertex) # Mean over batch
			loss = (1-lossRatio)*lossSeg + lossRatio*lossVertex

			# Update moving average loss
			lossSegTotal = (lossSegTotal*idx + lossSeg.item())/(idx+1)
			lossVertexTotal = (lossVertexTotal*idx + lossVertex.item())/(idx+1)
			lossTotal = (lossTotal*idx + loss.item())/(idx+1)

	return lossTotal, lossVertexTotal, lossSegTotal























# MAIN SCRIPT
###############################################


# Create the model
network = Resnet18_8s(ver_dim=nKeypoints*2, seg_dim=2)
network = DataParallel(network).cuda()

# Initialize the optimizer
optimizer = Adam(network.parameters(), lr=learningRate)

# Create the training dataloader
classIdx = classNameToIdx[className]
#trainSet = IcaDataset(classIdx, trainScenesDir, formats, keypoints, skipInvisible=skipInvisible)
trainSet = IcaDataset(classIdx, trainScenesDir, formats, keypoints, visibilityThreshold=visThreshold)
if trainSet.len <= 0:
	print('The training dataset is empty. Exiting program.')
	exit()
trainSampler = RandomSampler(trainSet)
trainBatchSampler = BatchSampler(trainSampler, batchSize, drop_last=True)  # Torch sampler
trainLoader = DataLoader(trainSet, batch_sampler=trainBatchSampler, num_workers=8)
print('A total of '+str(len(trainSet))+' training images have been found.')

# Create the validation dataloader
valSet = IcaDataset(classIdx, valScenesDir, formats, keypoints)
if valSet.len <= 0:
	print('The validation dataset is empty. Exiting program.')
	exit()
valSampler = RandomSampler(valSet)
valBatchSampler = BatchSampler(valSampler, batchSize, drop_last=True)  # Torch sampler
valLoader = DataLoader(valSet, batch_sampler=valBatchSampler, num_workers=8)
print('A total of '+str(len(valSet))+' validation images have been found.')

# Initialize loss lists
trainLossTotal = []
trainLossVertexTotal = []
trainLossSegTotal = []
valLossTotal = []
valLossVertexTotal = []
valLossSegTotal = []

# Train the model
valLossBest = 1.7976931348623157e+30
valLossEarlyStoppingBest = valLossBest
nEpochSinceImprovement = 0
startingEpoch = 0

# Load states (if resuming)
if resumeTraining:
	print('Attempting to resume training from checkpoint at ' + checkpointPath)
	if os.path.isfile(checkpointPath):
		checkpoint = torch.load(checkpointPath)
		network.load_state_dict(checkpoint['network'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		trainLossTotal, trainLossVertexTotal, trainLossSegTotal, valLossTotal, valLossVertexTotal, valLossSegTotal = checkpoint['loss']
		startingEpoch = checkpoint['epoch']+1
		valLossBest = checkpoint['valLossBest']
		valLossEarlyStoppingBest = checkpoint['valLossEarlyStoppingBest']
		nEpochSinceImprovement = checkpoint['nEpochSinceImprovement']
	else:
		print('No checkpoint found at ' + checkpointPath + ', training network from scratch. Press enter to continue.')

for iEpoch in range(startingEpoch, nEpochs):

	print()
	print('################################################')
	print('Starting epoch #'+str(iEpoch)+' out of '+str(nEpochs-1))
	print('################################################')
	print()

	# Start epoch timer
	tTrainEpochStart = time.time()

	# Train the network
	tTrainEpochStart = time.time()
	trainLoss, trainLossVertex, trainLossSeg = train(network, trainLoader, optimizer, iterPause=30)
	trainLossTotal.append(trainLoss)
	trainLossVertexTotal.append(trainLossVertex)
	trainLossSegTotal.append(trainLossSeg)
	print()
	print('Training loss at the end of epoch {}: {}'.format(str(iEpoch), trainLoss))
	print('Time elapsed: {}'.format(time.time() - tTrainEpochStart))
	print('-----------------------------------------------------')
	print()

	# Clear cuda cache to make room for validation (Not sure if this does anything)
	torch.cuda.empty_cache() 

	# Validate the model
	tValEpochStart = time.time()
	valLoss, valLossVertex, valLossSeg = validate(network, valLoader)
	valLossTotal.append(valLoss)
	valLossVertexTotal.append(valLossVertex)
	valLossSegTotal.append(valLossSeg)
	print()
	print('Validation loss at the end of epoch {}: {}'.format(str(iEpoch), valLoss))
	print('Time elapsed: {}'.format(time.time() - tValEpochStart))
	print('-----------------------------------------------------')
	print()

	# Print some statistics
	print('All losses of epoch {}:'.format(iEpoch))
	print('trainLossTotal  ',trainLossTotal[iEpoch])
	print('trainLossVertexTotal  ',trainLossVertexTotal[iEpoch])
	print('trainLossSegTotal  ',trainLossSegTotal[iEpoch])
	print('valLossTotal  ',valLossTotal[iEpoch])
	print('valLossVertexTotal  ',valLossVertexTotal[iEpoch])
	print('valLossSegTotal  ',valLossSegTotal[iEpoch])
	print()
	print('Epoch {} took a total of {} seconds.'.format(iEpoch, time.time() - tTrainEpochStart))
	print('{} epochs since last improvement, early stopping after {}.'.format(nEpochSinceImprovement, patience-nEpochSinceImprovement))
	print('-----------------------------------------------------')
	print()

	# Save the checkpoint
	checkpoint = {
		'epoch': iEpoch,
		'network': network.state_dict(),
		'optimizer': optimizer.state_dict(),
		'loss': (trainLossTotal, trainLossVertexTotal, trainLossSegTotal, valLossTotal, valLossVertexTotal, valLossSegTotal),
		'epoch': iEpoch,
		'valLossBest': valLossBest,
		'valLossEarlyStoppingBest': valLossEarlyStoppingBest,
		'nEpochSinceImprovement': nEpochSinceImprovement
		}
	torch.save(checkpoint, checkpointPath)

	# Save the best model (TODO: Bake the datetime into filename)
	if valLoss < valLossBest:
		valLossBest = valLoss
		torch.save(network.state_dict(), bestNetPath)
		print('Saved best model at the end of epoch {}.'.format(iEpoch))
		if valLossBest < valLossEarlyStoppingBest*(1-minDeltaFactor):
			valLossEarlyStoppingBest = valLossBest
			nEpochSinceImprovement = 0
	nEpochSinceImprovement += 1

	# Check early stopping condition
	if nEpochSinceImprovement > patience:
		print('EARLY STOPPING: Stopped at epoch {} after {} epochs without improvement'.format(iEpoch, patience))
		break 



#End epoch loop
if iEpoch == nEpochs:
	print('Stopped at epoch {} which was the final epoch.'.format(iEpoch))
