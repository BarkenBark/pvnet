# Add project root to path
import os
import sys
sys.path.append('.')
sys.path.append('..')

# Own modules
from lib.ica.utils import parse_3D_keypoints
from lib.ica.run_utils import run_network

# Pvnet modules
from lib.networks.model_repository import Resnet18_8s
from lib.utils.data_utils import read_rgb_np
from lib.utils.draw_utils import imagenet_to_uint8, visualize_bounding_box, visualize_mask, visualize_vertex_field, visualize_overlap_mask, visualize_points, visualize_hypothesis
#import lib.ransac_voting_gpu_layer.ransac_voting as ransac_voting
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_hypothesis, ransac_voting_center


# Other modules
import torch
from PIL import Image
import numpy as np
from torch.nn import DataParallel
import torchvision.transforms as transforms # Used for converting input rgb to tensor, and normalizing to values suitable for ImageNet-trained models

# GLOBALS
from lib.ica.globals import imageNetMean, imageNetStd




# SETTINGS
#############################

dataDir = '/var/www/webdav/Data/ICA/Scene1/pvnet'
className = 'tval'
rgbIdx = 1

rgbDir = os.path.join(dataDir, 'rgb')
rgbPath = rgbDir + '/' + str(rgbIdx) + '.' + 'jpg' 







# Visualize results
###############################

# Calculate network output
segPred, verPred = run_network(dataDir, className, rgbIdx=rgbIdx)

# Segmentation
img = Image.open(rgbPath)
img = np.array(img)
visualize_overlap_mask(img, np.squeeze(segPred.cpu().detach().numpy()), None)

# Vertex field
verWeight = segPred.float().cpu().detach()
verWeight = np.argmax(verWeight, axis=1)
verWeight = verWeight[None,:,:,:]
visualize_vertex_field(verPred, verWeight, keypointIdx=0)

# Ransac Hypotheses
_,nKeypoints_x2,h,w = verPred.shape
nKeypoints = nKeypoints_x2//2
verPredAlt = verPred.reshape([1,nKeypoints,2,h,w]).permute([0,3,4,1,2])
hypothesisPoints, inlierCounts = ransac_voting_hypothesis(verWeight.squeeze(0).cuda(), verPredAlt, 1024, inlier_thresh=0.999)
visualize_hypothesis(img[None,:,:,:], hypothesisPoints.cpu().detach().numpy(), inlierCounts.cpu().detach().numpy(), None)

# Save hypothesis points and scores
hypPath = os.path.join('.', 'hypothesisPoints.npy')
scorePath = os.path.join('.', 'inlierCounts.npy')
np.save(hypPath, hypothesisPoints.cpu().detach().numpy())
np.save(scorePath, inlierCounts.cpu().detach().numpy())

non_maximum_supression_np(detections, scores, similarityThreshold, similarityFun, scoreThreshold=None, neighborThreshold=None)

