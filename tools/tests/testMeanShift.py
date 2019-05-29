import sys
sys.path.append('.')
sys.path.append('..')

from numpy import array
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import pickle
import math
from mpl_toolkits.mplot3d import Axes3D

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R






def rotationMatrixToEulerAngles(R) :
 
    #assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])








with open('./poseGlobal.pickle','rb') as file:
    poseGlobal = pickle.load(file)

centers = poseGlobal[:,:,-1]
centerNorms = np.linalg.norm(centers, axis=1)
nanIdx = np.where(np.isnan(centerNorms))
centerNorms[nanIdx] = -1
centers = centers[(centerNorms < 100) & (centerNorms > 0)]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(centers[:,0],centers[:,1],centers[:,2],c='red')
ax.axis('equal')
plt.show()



res=200
bandwidthList = list(np.linspace(0.01,0.045,res))
nCenters = np.zeros((res))
i=0
for bandwidth in bandwidthList:
    meanShiftClusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=5, cluster_all=False)
    wasError = False
    try:
        meanShiftClusterer.fit(centers)
    except ValueError:
        print('No point for bandwidth {}'.format(bandwidth))
        wasError = True
    if not wasError:
        clusterCenters = meanShiftClusterer.cluster_centers_
        nCenters[i] = clusterCenters.shape[0]
    else:
        nCenters[i] = 0
    i+=1

plt.plot(np.array(bandwidthList), nCenters)
plt.show()











########### Now test in 6D space ####################
######################################################
nInstances = poseGlobal.shape[0]
poseAlt = np.zeros((nInstances, 6))
for i in range(nInstances):
    pose = poseGlobal[i]
    R = pose[0:3,0:3]
    t = pose[:,-1]
    eul = rotationMatrixToEulerAngles(R)
    poseAlt[i,:] = np.concatenate((eul/100,t))

print('poseAlt  ',poseAlt)
centers = poseAlt[:,3:6]
centerNorms = np.linalg.norm(centers, axis=1)
nanIdx = np.where(np.isnan(centerNorms))
centerNorms[nanIdx] = -1
poseAlt = poseAlt[(centerNorms < 100) & (centerNorms > 0), :]

# Meanshift
resBand = 50
binFreqStart = 3
binFreqEnd = 7
bandwidthList = list(np.linspace(0.01,0.045,resBand))
freqList = list(range(binFreqStart, binFreqEnd+1))

clusterPosesAltHist = []
nPoses = []
j=0
for min_bin_freq in freqList:
    i=0
    for bandwidth in bandwidthList:
        meanShiftClusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=min_bin_freq, cluster_all=False)
        wasError = False
        try:
            meanShiftClusterer.fit(poseAlt)
        except ValueError:
            print('No point for bandwidth {}'.format(bandwidth))
            wasError = True
        if not wasError:
            clusterPosesAlt = meanShiftClusterer.cluster_centers_
            clusterPosesAltHist.append(clusterPosesAlt)
            nPoses.append(clusterPosesAlt.shape[0])

        i+=1
    j+=1

bestNPoses = max(set(nPoses), key=nPoses.count) # Will be a number which decides our final guess for nInstances in the scenes
clusterPosesAlt = [clusterPosesAltHist[i] for i in range(len(clusterPosesAltHist)) if nPoses[i]==bestNPoses]
clusterPosesAlt = array(clusterPosesAlt)

print('clusterPosesAlt.shape  ',clusterPosesAlt.shape)

plotPoints = clusterPosesAlt[:,:,3:]
plotPoints1 = plotPoints[:,0]
plotPoints2 = plotPoints[:,1]
#plotPoints3 = plotPoints[:,2]

print('plotPoints1.shape  ', plotPoints1.shape)
print('plotPoints2.shape  ', plotPoints2.shape)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(plotPoints1[:,0],plotPoints1[:,1],plotPoints1[:,2],c='red')
ax.scatter(plotPoints2[:,0],plotPoints2[:,1],plotPoints2[:,2],c='blue')
#ax.scatter(plotPoints3[:,0],plotPoints3[:,1],plotPoints3[:,2],c='green')
ax.axis('equal')
plt.show()