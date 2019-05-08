#%%
import os 
import sys
import yaml
sys.path.append('.')
sys.path.append('..')

from lib.ica.utils import *
from lib.ica.run_utils import calculate_epipolar_lines
from lib.utils.draw_utils import visualize_lines

import pickle
import yaml
from numpy import array
import numpy as np
from numpy.linalg import inv

dataDir = '/var/www/webdav/Data/ICA/Scene1/pvnet'
className = 'tval'
paths = get_work_paths(dataDir, className)
#poseOutDir = os.path.join(dataDir, 'poseannotation_out')
#posesPath = os.path.join(paths['poseOutDir'], 'poses.yml')
#poseData = yaml.load(open(paths['posesPath'], 'r'))

#def getCam(iView):
#    R = array(poseData[iView][0]['cam_R_m2c']).reshape((3,3))
#    t = array(poseData[iView][0]['cam_t_m2c']).reshape((3,1))
#    pose = np.hstack((R, t))
#    return pose

#cameras = [parse_pose(poseData, iView, 0) for iView in range(991)]
#[cam @ cam[0] for cam in cameras]

#trainRgbDir = os.path.join(dataDir, 'rgb')
#cameraPath = trainRgbDir + '/camera.yml'
#cameraData = yaml.load(open(cameraPath, 'r'))
motion = parse_motion(os.path.join(paths['poseOutDir'], 'motionScaled.txt'))
motion = motion[0:30]
K = parse_inner_parameters(paths['cameraPath'])
#K = np.zeros((3,3))
#K[0,0] = cameraData['fx']
#K[1,1] = cameraData['fy']
#K[0,2] = cameraData['cx']
#K[1,2] = cameraData['cy']

with open('points2D.pickle', 'rb') as file:
    points2D = pickle.load(file)

with open('covariance.pickle', 'rb') as file:
	covariance = pickle.load(file)

# Pre-process
points2D_n = normalize_list(points2D, K)
#%% Experiment: Plot the epipolar lines of all other cameras onto camera 0

keypointIdx = 0
nCameras = len(points2D_n)
thisKeypoints = [keypoints[:, keypointIdx, :] if keypoints is not None else None for iCamera, keypoints in enumerate(points2D_n)]

#%%
epiLines = calculate_epipolar_lines(motion, thisKeypoints, projectionIdx=0)
epiLinesPlot = np.empty((3,0))
for i, lines in enumerate(epiLines):
	if lines is not None:
		epiLinesPlot = np.append(epiLinesPlot, lines, axis=1)

#%%
visualize_lines(None, epiLinesPlot, [-1,1,-1,1])

























