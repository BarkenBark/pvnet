# IMPORT STATEMENTS
#################################################
import os
import sys
import time
sys.path.append('.')
sys.path.append('..')

# Import own modules
from lib.ica.utils import * # Includes parse_3D_keypoints, pflat, pextend etc. 

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
from PIL import Image
from numpy import eye, unique
import yaml

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd










# Argument variables (TODO: Parse from argument or settings file)
#################################
# Paths
dataDir = '/var/www/webdav/Data/ICA/Scenes/Train/Scene1/pvnet'
className = 'seltin' # MAYBE REPLACE WITH CLASSIDX NUMBER?
# Training parameters
nEpochs = 100
resumeTraining = False
batchSize = 16
lossRatio = 0.5 # Weight ratio of vertex loss and segmentation loss (1 is all vertex, 0 is all segmentation)

# Optimization parameters
learningRate = 1e-3

















# CLASS DEFINITIONS
###############################

class IcaDataset(Dataset):
	def __init__(self, classIdx, nLeadingZerosFormat, rgbDir, rgbFormat, segDir, segFormat, posesPath, keypoints, K):

		# Trivial class member initilizations
		self.classIdx = classIdx # NOTE: Indexed from 1 to nClasses
		self.nLeadingZerosFormat = nLeadingZerosFormat
		self.rgbDir = rgbDir
		self.rgbFormat = rgbFormat
		self.segDir = segDir
		self.segFormat = segFormat
		self.keypoints = keypoints
		self.poseData = yaml.load(open(posesPath, 'r'))
		self.K = K

		# Define transformation to normalize input images with ImageNet values
		self.test_img_transforms = transforms.Compose([
		    transforms.ToTensor(), # if image.dtype is np.uint8, then it will be divided by 255
		    transforms.Normalize(mean=imageNetMean,
		                         std=imageNetStd)
		])

		# # Check number of images to determine self.len
		self.len = find_nbr_of_files(rgbDir, format=rgbFormat)
		print('Found ' + str(self.len) + ' images in ' + rgbDir)

		# Assert conditions
		assert(self.len == find_nbr_of_files(segDir, format=segFormat))

		# Print warnings
		print('WARNING: Assuming that the instance order of instanceIdx is the same as the instance order in poses.yml')



	def __getitem__(self, index):
		# OUTPUTS:
		# rgb - input rgb tensor [3, height, width]
		# mask - ground truth segmentation tensor [2, height, width]
		# ver - ground truth vertex tensor [2*nKeypoints, height, width]
		# verWeights - ground truth binary weight tensor [_, height, width]

		# RGB
		rgbPath = self.rgbDir + '/' + str(index+1) + '.' + self.rgbFormat # TO-DO: Change s.t. we can read from several different folders
		rgb = read_rgb_np(rgbPath) #  Np array
		rgb = self.test_img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))
		_, height, width = rgb.shape

		# Mask
		segPath = self.segDir + '/' + str(index+1).zfill(self.nLeadingZerosFormat) + '.' + self.segFormat
		segImg = np.asarray(Image.open(segPath))
		instanceIdxPath = self.segDir + '/' + 'classindex.txt'
		instanceIdx = np.loadtxt(instanceIdxPath, delimiter=',')
		idxMatch = (instanceIdx == self.classIdx).astype(int)
		idxMatch = np.concatenate(([0], idxMatch)) # Add 'no-class' idx
		mask = torch.tensor(idxMatch[segImg], dtype=torch.int64)

		# Vertex
		nKeypoints = (self.keypoints).shape[1]
		ver = np.zeros([height,width,nKeypoints*2],np.float32)
		instanceSegImg = mask.numpy()*segImg
		nInstances = sum(idxMatch)

		idx = [i for i,j in enumerate(instanceIdx) if j==self.classIdx] 
		poses = [parse_pose(self.poseData, index, iPose) for iPose in idx]

		for iInstance in range(nInstances):
			thisMask = instanceSegImg == idx[iInstance]+1
			keypointsProjected = self.K @ (poses[iInstance] @ pextend(self.keypoints, vecType='col'))
			keypointsProjected = pflat(keypointsProjected)
			ver = ver + compute_vertex_hcoords(thisMask, keypointsProjected.T)

		ver=torch.tensor(ver, dtype=torch.float32).permute(2, 0, 1)
		verWeights=mask.unsqueeze(0).float()

		return rgb, mask, ver, verWeights

	def __len__(self):
		return self.len















# DEFINE VARIABLES (TODO: Move some of these to a settings file, let some be command arguments)
############################################

# TO-DO: Replace with get_work_paths
# Implicit paths 
paths = get_work_paths(dataDir, className=className)

# If networkDir does not exist, create it
try:
	os.mkdir(os.path.join(paths['networkDir'], className))
except FileExistsError:
	print('Network directory already exists.')

# Implicit variables
####################################
rgbFormat = 'jpg' # (TODO: FORMAT OF FIRST IMAGE IN RGB-DIRECTORY)
segFormat = 'png' # (TODO: FORMAT OF FIRST IMAGE IN SEG-DIRECTORY)
nLeadingZerosFormat = 5 # (TODO: FORMAT OF FIRST IMAGE IN SEG-DIRECTORY. ASSERT SAME AS RGB)
keypoints = parse_3D_keypoints(paths['keypointsPath'], addCenter=True)
nKeypoints = keypoints.shape[1]
classNameToIdx, _ = create_class_idx_dict(paths['modelDir'])
K = parse_inner_parameters(paths['rgbDir'] + '/camera.yml') # Camera inner parameters














# MAIN SCRIPT
###############################################


# Create the model
network = Resnet18_8s(ver_dim=nKeypoints*2, seg_dim=2)
network = DataParallel(network).cuda()
if resumeTraining:
	print('Attempting to load weights from ' + paths['networkPatḧ́'])
	if os.path.isfile(paths['networkPatḧ́']):
		network.load_state_dict(torch.load(paths['networkPatḧ́']))
	else:
		input('No network found at ' + paths['networkPatḧ́'] + ', training network from scratch. Press enter to continue.')

# Create the dataloader
classIdx = classNameToIdx[className]
trainSet = IcaDataset(classIdx, nLeadingZerosFormat, paths['rgbDir'], rgbFormat, paths['segDir'], segFormat, paths['posesPath'], keypoints, K) # Torch dataset
trainSampler = RandomSampler(trainSet)
trainBatchSampler = BatchSampler(trainSampler, batchSize, drop_last=True)  # Torch sampler
trainLoader = DataLoader(trainSet, batch_sampler=trainBatchSampler, num_workers=8)

# Initialize the optimizer
optimizer = Adam(network.parameters(), lr=learningRate)


# Train the model
nIterations = len(trainSet) // batchSize
for iEpoch in range(nEpochs):
	print('Starting epoch #'+str(iEpoch+1)+' out of '+str(nEpochs))
	tEpochStart = time.time()
	for idx, data in enumerate(trainLoader):
		
		# Start training loop iteration timer
		tTrainingLoopStart = time.time()

		# Extract data
		tExtractDataStart = time.time()
		image, maskGT, vertexGT, vertexWeightsGT = [d.cuda() for d in data]
		tExtractDataElapsed = time.time() - tExtractDataStart
		
		# DEBUGGING: Print shapes and dtypes of the tensors
		#print_attributes('dtype', 'shape', image=image, maskGT=maskGT, vertexGT=vertexGT, vertexWeightsGT=vertexWeightsGT)

		# REMOVE:
		#visualize_vertex_field(vertexGT.clone(), vertexWeightsGT.clone(), keypointIdx=2)

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

		# Update weights
		tUpdateWeightStart = time.time()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		tUpdateWeightElapsed = time.time() - tUpdateWeightStart

		# Print training loop iteration time
		tTrainingLoopElapsed = time.time() - tTrainingLoopStart
		if (idx % 10)==0:
			# print('Extracting data took ' + str(tExtractDataElapsed) + ' seconds.')
			# print('Forward propagating took ' + str(tForwardPropElapsed) + ' seconds.')
			# print('Computing loss took ' + str(tComputeLossElapsed) + ' seconds.')
			# print('Updating weights took ' + str(tUpdateWeightElapsed) + ' seconds.')
			# print('Individual steps took at total of {} seconds.'.format(tExtractDataElapsed+tForwardPropElapsed+tComputeLossElapsed+tUpdateWeightElapsed))
			print('In total, training loop iteration {}/{} took {} seconds.'.format(idx, nIterations ,tTrainingLoopElapsed))
			print()

	# Save the model (TODO: Bake the datetime into filename)
	torch.save(network.state_dict(), paths['networkPath'])

	# Retrieve time
	tEpochElapsed = time.time() - tEpochStart

	# Print status
	print()
	print('Loss at the end of epoch {}: {}'.format(str(iEpoch), loss.item()))
	print('Time elapsed: {}'.format(tEpochElapsed))
	print('-----------------------------------------------------')

	