import pickle
import matplotlib.pyplot as plt
from PIL import Image
from numpy import genfromtxt,array,reshape,hstack,vstack,all,any,sum,zeros, swapaxes,unique,concatenate,repeat,stack,tensordot,pad,linspace,mean,argmax
import numpy as np
import cv2
import time
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
import itertools
import time

def nms(detections, scores, similarityThresholdList, similarityFun, scoreThresholdList=None, neighborThresholdList=None):
    # Input: 
    # detections - (1+N)D-np.ndarray of data point candidates detected by algorithm, e.g. bounding boxes for object detection. First dimension is the batch dimension.
    # scores - 1D-np.ndarray of scores associated with the detections. 
    # similarityThresholdList - list of values for when to discard detections
    #   too similar to the currently top-scoring detection
    # similarityFun - function which returns an np.ndarray of similarity measures between current detection (arg1) and other detections (arg2)

    # Output:
    # filteredDetectionIdx - list of indices of 'detections' that remain after NMS
    assert(scores.shape[0] == detections.shape[0])

    # If similarityThresholdList,scoreThresholdList or neighborThresholdList is
    #   a single scalar, change it to be withinin a list
    if not type(similarityThresholdList)==list:
        similarityThresholdList = [similarityThresholdList]
        
    if not type(scoreThresholdList)==list:
        scoreThresholdList = [scoreThresholdList]
        
    if not type(neighborThresholdList)==list:
        neighborThresholdList = [neighborThresholdList]
    
    nSimVals = len(similarityThresholdList)
    nScoreVals = len(scoreThresholdList)
    nNeighVals = len(neighborThresholdList)
    
    # filteredDetectionIdx - multidepth list containing the filtered detections.
    #   It is indexed according to filteredDetectionIdx[iSim][iScore][iNeigh] = [idx1,idx2,...]
    #       where iSim, iScore, iNeigh is a triplet of threshold values
    filteredDetectionIdx = [[[[] for j in range(nNeighVals)] for i in range(nScoreVals)] for k in range(nSimVals)]
    
    # Run nms for each unique triplet of threshold values
    for iSim in range(nSimVals):
        for iScore in range(nScoreVals):
            for iNeigh in range(nNeighVals):
                similarityThreshold = similarityThresholdList[iSim]
                scoreThreshold = scoreThresholdList[iScore]
                neighborThreshold = neighborThresholdList[iNeigh]                
                remainingIdx = scores.argsort()[::-1] # Scores for detections in descending order                        
                while remainingIdx.shape[0] > 0:
                    idx = remainingIdx[0]
            
                    # If the score is too low, return
                    if scoreThreshold is not None:
                        if scores[idx] < scoreThreshold:
                            break
            
                    # Determine remaining detections
                    thisDetection = detections[idx]
                    remainingDetections = detections[remainingIdx[1:]]
                    # Calculate the similiarity between thisDetection and all other remaining detections
                    sim = similarityFun(thisDetection, remainingDetections)
            
                    # Remove neighbors (detections too similar to thisDetection) from rotation 
                    # keepIdx = np.where(sim <= threshold)
                    # remainingIdx = remainingIdx[keepIdx[0] + 1]
                    removeIdx = np.where(sim > similarityThreshold)[0] + 1
                    removeIdx = np.insert(removeIdx, 0, 0) # Prepend with 0 to remove current idx
                    remainingIdx = np.delete(remainingIdx, removeIdx)
            
                    # If this detection does not have enough neighbors (detections similar enough), consider it a false positive and discard it
                    if neighborThreshold is not None:
                        nNeighbors = removeIdx.shape[0] - 1
                        if nNeighbors > neighborThreshold:
                            filteredDetectionIdx[iSim][iScore][iNeigh].append(idx)
                    else:
                        filteredDetectionIdx[iSim][iScore][iNeigh].append(idx)
    
    # If single values were given for all thresholds, return a list without
    #   multiple depths for backwards compability
    if not type(similarityThresholdList)==list and \
        not type(scoreThresholdList)==list and \
            not type(neighborThresholdList)==list:
                filteredDetectionIdx = filteredDetectionIdx[0][0][0]
    return filteredDetectionIdx

def getCenters(detections):
    # detections.shape==(b,3,4) or (3,4) where b is batch size
    # If batch size > 1, calculate center with tensor multiplication
    
    if len(detections.shape) == 3:
        centers = detections[:,:,-1]
    else:
        centers = detections[:,-1]
    
# =============================================================================
#     if detections.shape[0] > 1 and len(detections.shape) == 3:
#         detections = np.swapaxes(detections,0,2)
#         centers = np.sum(detections[0:3,:,:] * detections[-1:,:,:],axis=1)
#     else:
#         centers = detections.squeeze()[:,0:3] @ detections.squeeze()[:,-1:]
# =============================================================================
    
    return centers

def centerSimilarity(thisDetection, detections):
    # Get center and centers
    centers = getCenters(detections)
    thisCenter = getCenters(thisDetection)
    
    # Calulate similarity as the inverse of distance
    distances = norm(centers - thisCenter,axis=1)
    similarity = 1/distances
    return similarity

def pextend(x):
    return np.vstack((x,np.ones((1,x.shape[1]))))

def pflat(x):
    return x[0:2,:]/x[-1:,:]

def transformMultiplePoses(poses1, camera1, cameras):
    poses1 = poseArray
    nPoses = poses1.shape[0]
    poseTransform  = concatenate((poses1,repeat(array([0,0,0,1])[None,None,:],nPoses,axis=0)),axis=1)
    camerasArr = stack(cameras)    
    posesInViews = swapaxes(tensordot(camerasArr,poseTransform, axes=([2],[1])),1,2)
    return posesInViews
    
    
def transformPose(poseInCameraX, cameraX, cameraY):
    poseTransform = np.vstack((poseInCameraX ,np.array([0, 0, 0, 1])[None,:]))

    RCam = cameraX[:,0:3]
    tCam = cameraX[:,-1:]
    PCam = np.hstack((RCam.T,-RCam.T @ tCam))

    camTransform  = np.vstack((PCam,np.array([0, 0, 0, 1])[None,:]))

    poseInCameraY = cameraY @ camTransform @ poseTransform
    return poseInCameraY

def calculatePose(points2D, points3D):
    # Input: 
    # points2D - ndarray, shape=(nInstances,2,nKeypoints)
    # points3D - ndarray, shape=(3,nKeypoints)
    
    # Output:
    # poses - ndarray, shape=(nInstances,3,4)
    
    poses = zeros((len(points2D),3,4))
    for iPoint, point in enumerate(points2D):
        # Solve pnp problem
        poseParam = cv2.solvePnP(objectPoints = points3D.T, imagePoints = point.T[:,None],\
                                     cameraMatrix = camera_matrix, distCoeffs = None,flags = cv2.SOLVEPNP_EPNP)
        
        # Extract pose related data and calculate pose
        R = cv2.Rodrigues(poseParam[1])[0]
        t = poseParam[2]
        pose = np.hstack((R,t))
        poses[iPoint] = pose
    return poses

# Do not use
def calculatePose1(points2D, camera, points3D):
    poseList = []
    for point in points2D:
        poseParam = cv2.solvePnP(objectPoints = points3D.T, imagePoints = point.T[:,None],\
                                     cameraMatrix = camera_matrix, distCoeffs = None,flags = cv2.SOLVEPNP_EPNP)
        R = cv2.Rodrigues(poseParam[1])[0]
        t = poseParam[2]
        pose = np.hstack((R,t))
        pose1 = transformPose(pose, camera, cameras[0])
        poseList.append(pose1)       
    return poseList

#%%

# Read object keypoints from file
points3D = genfromtxt('/home/andreas/Studies/MasterArbeteLokal/PoseAnnotation/Tests/App/Ginput within subplot/Scene2/models/tval_keypoints.txt', delimiter=',')
points3D = np.hstack((np.zeros((3,1)),points3D))
points3DHomo = pextend(points3D)
nKeypoints = points3D.shape[1]

# Read 2D keypoints (algorithm output) from file
with open('/home/andreas/Downloads/points2D.pickle','rb') as file:
    points2D = pickle.load(file)
nViews = len(points2D)

# Check which views has any detections
hasInstances = [type(point) != type(None) for point in points2D]

# Multiply points with 2 to be compatible with andys rgs'b (remove later) and 
#   change shape (n,11,2) -> (n,2,11)
points2D = [swapaxes(2*points2D[iView],1,2) if hasInstances[iView] else None for iView in range(nViews)]

# Read camera motion from file
cameras = genfromtxt('/home/andreas/Downloads/motionScaled.txt', delimiter=' ')
cameras = [np.hstack((np.reshape(row[0:9],(3,3)),row[9:,None])) for row in cameras]

# Read 2D keypoints cov (algorithm output) from file
with open('/home/andreas/Downloads/covariance.pickle','rb') as file:
    cov = pickle.load(file)

camera_matrix = array([[929.4780, 0, 640.0000],[0, 929.4780, 360.0000],[0, 0, 1.0000]])

#%%
# Solve motion scale idea:
# Optimize somthing to match poses between all images as a function of
# scale only. Maybe ransac a pose in two different views and minimal solve to determine
# the optimal scale to make them match. Look at all other matches and give a matching
# score for this particular scale value.

# Improvements:
# Look at 3d ray distance instead of 2d distance
# Change similarity function to e.g. IoU
# Parameter sweep over nms (in progress)

# Threshold for reprojection distance between hypothesis and observed points
#   for being an inlier

inlierThresh = 7

def generateRescaledCameras(cameras, scale):
    rescaleMat = concatenate((np.ones((3,3)),np.ones((3,1))*scale),axis=1)
    camerasScaled = [rescaleMat*camera for camera in cameras]
    return camerasScaled, scale

def generateRandomlyRescaledCameras(cameras, scaleRange):
    randomScale = scaleRange[0] + (scaleRange[1] - scaleRange[0])*np.random.rand(1)
    camerasScaled, scale = generateRescaledCameras(cameras, randomScale)
    return camerasScaled, scale

def getSelectedData(nViewSkips=1, *args):
    # Return every 'nViewSkips' datapoints for all objects in args.
    #   Requires the length of the objects in args to have same size 
    #     along first dimension/axis
    nViews = len(args[0])
    
    # Set how many wiews to skip, i.e. use every 'nViewSkips' cameras
    selectedViewsRange = range(0,nViews,nViewSkips)
    selectedViewsSlice = slice(0,nViews,nViewSkips)
    selectedData = tuple([arg[selectedViewsSlice] if type(arg) == list else arg[selectedViewsRange] for arg in args])
    return selectedData


def getPoseHypotheses(cameras,points2D,points3D,nViewSkips=1):
    nViews = len(cameras)
    
    # Create list containing ndarray's of all poses in each selected view
    posesList = [calculate_pose(points2D[iView], points3D) for iView in range(nViews)]
    
    # Create list containing ndarray's of all poses in each selected view transformed into first camera
    poseListFirstCam = [[transform_pose(posesList[iView][iInstance], cameras[iView], cameras[0]) for iInstance in range(len(posesList[iView]))] for iView in range(nViews)]
    
    # Create array of all poses in first camera shape=(nPoses,3,4), and remove doubles
    poseArray = array(list(itertools.chain.from_iterable(poseListFirstCam)))
    poseArray = unique(poseArray,axis=0)
    
    return poseArray

def getNumInliers(poseArray, cameras, points2D, points3DHomo, camera_matrix, maxNbrInstances):
    nViews = len(cameras)
    nKeypoints = points3DHomo.shape[1]
    # Calculate the maximum number of detection instances in a single image
    maxNbrInstances = max([points2D[iView].shape[0] for iView in range(nViews)])
    
    # Transform each pose to all selected cameras
    posesInViews = transformMultiplePoses(poseArray, cameras[0], cameras)
    
    # Calculate 3D points for all poses in all views
    pointsInViews = tensordot(posesInViews,points3DHomo,axes=([3],[0]))
    cameraPointsInViews = swapaxes(tensordot(pointsInViews,camera_matrix,axes=([2],[1])),2,3)
    
    # Calculate 2D points for all poses in all cameras (project 3D points)
    projPointsInViews = (cameraPointsInViews[:,:,0:2,:]/cameraPointsInViews[:,:,-1:,:])
    
    # Pad the views which have a smaller number of detected instances than the
    #   maximum detected with empty projections. This is to be able to convert to an array.
    zeroArr = zeros((maxNbrInstances,2,nKeypoints))
    points2DPadded=[zeroArr if not hasInstances[iView] \
          else pad(points2D[iView],((0,maxNbrInstances-points2D[iView].shape[0]),(0,0),(0,0)),'constant',constant_values=0) \
          for iView in range(nViews)]
    
    # Create an ndarray with shape=() containing the projected points in all selected views
    #   from all poses
    points2DArr = stack(points2DPadded)
    
    # Calculate the distances between the projected points and the NN output
    distances = norm((points2DArr[:,None] - projPointsInViews[:,:,None]),axis=3)
    
    # Calculate the inliers for each pose
    inliers = any(all(distances < inlierThresh,axis=3),axis=2)
    
    # Calculate the total number of inlier for each pose
    nInliers = sum(inliers, axis=0)
    return nInliers
        
    
def estimateCamerasScale(cameras,points2D,points3D,nViewSkips):

    maxInliers = -1
    for scale in linspace(0,100,1000):
        
        # Rescale camera motion with current scale estimate
        camerasScaled = generateRescaledCameras(cameras, scale)
        
        # Calculate pose hypotheses and their number of inliers
        poseArray, nInliers = getPoseHypotheses(camerasScaled)
        
        # Calculate the sum of all inliers across all poses
        totalInliers = sum(nInliers)
        
        # Find the scale s which maximizes the total number of inlier across all poses
        if totalInliers > maxInliers:
            maxInliers = totalInliers
            realScale = scale

    return

# Calculate the maximum number of detection instances in a single image
maxNbrInstances = max([points2D[iView].shape[0] for iView in range(nViews) if hasInstances[iView]])
camerasSelected, points2DSelected = getSelectedData(2, cameras, points2D)

print('Cameras true rescale factor: ', rescaleTrue)
print('Cameras estimate rescale factor: ', 1/realScale)

#%%

nSweeps = 30
similarityThresholdList = [1/distance for distance in linspace(0.02,0.08,nSweeps)]
similarityFun=centerSimilarity
scoreThresholdList = [score for score in linspace(1,200,nSweeps)]
neighborThresholdList = [neigh for neigh in linspace(1,200,nSweeps)]

#non_maximum_supression_np(poseArray, nInliers, similarityThreshold, similarityFun, scoreThreshold=40, neighborThreshold=20)
t = time.time()
detectedPosesIdx = nms(poseArray, nInliers, similarityThresholdList, similarityFun, scoreThresholdList, neighborThresholdList)
print('Time to run parameter sweep:  ', time.time() - t)
nDetectedPosesIdx = [len(detectedPosesIdx[x][y][z]) for x in range(nSweeps) for y in range(nSweeps) for z in range(nSweeps)]
nDetectedPoses = max(set(nDetectedPosesIdx), key=nDetectedPosesIdx.count)

maxDetections = -1
for nDetections in unique(nDetectedPosesIdx):
    detectedPosesIdxMatchNDetected = [detectedPosesIdx[x][y][z] for x in range(nSweeps) for y in range(nSweeps) for z in range(nSweeps) if len(detectedPosesIdx[x][y][z]) == nDetections]
    uniqueDetectedPoses = unique(detectedPosesIdxMatchNDetected, axis=0,return_counts=True)
    print(max(uniqueDetectedPoses[1]))
    if max(uniqueDetectedPoses[1]) > maxDetections:
        maxDetections = max(uniqueDetectedPoses[1])
        detectedPosesIdxFinal = uniqueDetectedPoses[0][argmax(uniqueDetectedPoses[1])]

detectedPosesIdxFinal

plt.close()
plt.plot(x,y)
plt.show()
print(detectedPosesIdx)

t = time.time()
# do stuff
elapsed = time.time() - t

plt.close()
for idx in range(len(detectedPosesIdxFinal)):
    image_name = "/home/andreas/Studies/MasterArbeteLokal/PoseAnnotation/Tests/App/Ginput within subplot/Scene2/rgb/" + str(1) + ".jpg"
    img = plt.imread(image_name)
    implot = plt.imshow(img)
    #plt.scatter(points2D[iView][0,0,:],points2D[iView][0,1,:])
    P = poseArray[detectedPosesIdxFinal[idx]]
    print('Pose ' + str(idx) + ' ', P )
    projPoints = pflat(camera_matrix @ P @ points3DHomo)
    plt.scatter(projPoints[0,:],projPoints[1,:])

centers = getCenters(poseArray)
centerInliers = centers[all(centers < 3,1)]

plt.close()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(centerInliers[:,0],centerInliers[:,1],centerInliers[:,2])
ax.axis('equal')
#%%
# TODO: padda points2D med nollor så kan göra array

#[points2D[iView]-projPointsInViews[iView] for iView in range(nViews)]

#poses = [transformPose(pose, cameras[sampleView], cameras[iView]) for iView in range(nViews)]
#projPoints = [pflat(camera_matrix @ pose @ points3DHomo) for pose in poses]

# Create placeholders
scores = zeros((nSamples,1))
detections = zeros((nSamples,3,4))
sampleViews = zeros((nSamples,1))

# Create and evaluate hypotheses
iSample = 0
while iSample < nSamples:
    # Sample a view that has an instance
    sampleView = np.random.choice(range(0,nViews,20), 1,replace=False).squeeze()
    while type(points2D[sampleView]) == type(None):
        sampleView = np.random.choice(nViews, 1,replace=False).squeeze()

    nInstancesInView = points2D[sampleView].shape[0]
    
    for iInstance in range(nInstancesInView):
        # Sample 4 keypoints (2D-3D correspondances) in the sampled view
        sampleKeypoints = np.random.choice(nKeypoints , 4,replace=False)
        points2DSample = points2D[sampleView][iInstance,:,sampleKeypoints].T
        points3DSample = points3D[:,sampleKeypoints]
        
        # Use minimal solver with the sampled keypoints and extract the pose
# =============================================================================
#         poseParam = cv2.solvePnP(objectPoints = points3DSample.T, imagePoints = points2DSample[:,None,:].T,\
#                                  cameraMatrix = camera_matrix, distCoeffs = None,flags = cv2.SOLVEPNP_P3P)
# =============================================================================
        
        poseParam = cv2.solvePnP(objectPoints = points3D.T, imagePoints = points2D[sampleView][iInstance].T[:,None],\
                                 cameraMatrix = camera_matrix, distCoeffs = None,flags = cv2.SOLVEPNP_EPNP)
        
        R = cv2.Rodrigues(poseParam[1])[0]
        t = poseParam[2]
        pose = np.hstack((R,t))
        pose1 = transformPose(pose, cameras[sampleView], cameras[0])
        
        # Calculate the pose for all other views and project 3D keypoints into all images
        poses = [transformPose(pose, cameras[sampleView], cameras[iView]) for iView in range(nViews)]
        projPoints = [pflat(camera_matrix @ pose @ points3DHomo) for pose in poses]
        
        # Create a boolean list inliers. If the distances between all 
        # detected (algorithm output) and projected keypoints for an instance
        # in a view are within the 'inlierThresh' it is considered an inlier.
        # One view can at most yield on inlier.
        inlierList = [False if type(points2D[iView]) == type(None) else\
                      any(all((norm(points2D[iView] - projPoints[iView][None],axis=1)<inlierThresh),axis=1))\
                      for iView in range(nViews)]
        
        # Save the number of inliers, the pose, and the sample view
        nInliers = sum(inlierList)
        scores[iSample] = nInliers
        detections[iSample] = pose1
        sampleViews[iSample] = sampleView
        iSample +=1
        if iSample == nSamples:
            break

# Remove all samples that does not have any inliers
hasInliers = (scores != 0).squeeze()
detections = detections[hasInliers]
sampleViews = sampleViews[hasInliers]
scores = scores[hasInliers].squeeze()

similarityThreshold=1/0.05
similarityFun = centerSimilarity

detectedPosesIdx = non_maximum_supression_np(detections, scores, similarityThreshold, similarityFun, scoreThreshold=nSamples/5, neighborThreshold=nSamples/10)
centers = getCenters(detections[detectedPosesIdx])
print(centers )

inliers = all(centers < 3,1)
centerInliers = centers[inliers]

