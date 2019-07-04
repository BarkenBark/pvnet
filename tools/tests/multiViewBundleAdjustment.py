def eulerAnglesToRotationMatrix(theta) :
    theta = theta[::-1]
     
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

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-3
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
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
    return np.array([z,y,x])



def error(imagePointsList, modelPointsList, eulPose, cameras):
    #cameras: list of cameras
    #imagePointsList: list of length nViews, containing arrays of varying shape = (nImagePointsx2)
    #modelPointsList: list of length nViews, containing arrays of varying shape = (nImagePointsx2)
    #eulpose: (6,) pose where rotationmatrix is replaced with euler angles
    nViews = np.size(cameras,axis=0)
    RPose = eulerAnglesToRotationMatrix(eulPose[0:3])
    tPose = eulPose[3:6]
    error = 0
    
    for iView in range (nViews):
        P = cameras[iView]
        R = P[:,0:3]
        t = P[:,3]
        imagePoints = imagePointsList[iView].T
        nImagePoints = imagePoints.shape[1]
        modelPoints = modelPointsList[iView].T
        for iImagePoint in range(nImagePoints):
            proj = R @ (RPose @ modelPoints[:,iImagePoint:iImagePoint+1]+tPose[:,None])+t[:,None]
            error = error + np.linalg.norm( [imagePoints[0,iImagePoint] - proj[0]/proj[2], imagePoints[1,iImagePoint] - proj[1]/proj[2] ])**2
            
    return error


# =============================================================================
# imagePointsList = [np.array([[-0.0637,0.2170,1],[-0.1002,0.2270,1],[-0.1522,0.0946,1],[-0.1832,0.0606,1],[-0.1747,0.1396,1],[-0.1252,0.0636,1],[-0.1207,0.1171,1]]),np.array([[0.4195,0.0093,1],[0.3801,0.0043,1],[0.4502,-0.2242,1],[0.4519,-0.1535,1],[0.5106,-0.2116,1],[0.5119,-0.1632,1]])]
# modelPointsList = [np.array([[-0.0161,0.0437,-0.0867],[0.0153,0.0438,-0.0850],[-0.0009,-0.0017,0.1114],[0.0135,-0.0224,0.1100],[0.0141,0.0199,0.1108],[-0.0175,-0.0169,0.1104],[-0.0176,0.0125,0.1109]]),np.array([[-0.0161,0.0437,-0.0867],[0.0153,0.0438,-0.0850],[0.0135,-0.0224,0.1100],[0.0141,0.0199,0.1108],[-0.0175,-0.0169,0.1104],[-0.0176,0.0125,0.1109]])]
# cameras = [np.array([[0.6763,0.3008,-0.6724,0.7985],[-0.7087,0.5145,-0.4827,0.2955],[0.2008,0.8030,0.5611,0.2399]]),
#            np.array([[0.8411,0.3174,-0.4380,0.9958],[-0.5391,0.5570,-0.6317,0.3421],[0.0434,0.7675,0.6396,0.1189]])]
# eulPose = np.array([1,2,3,0,0,1])
# 
# =============================================================================
    
def multiviewPoseEstimation(imagePointsList,modelPointsList,cameras,pose0=None):
    if pose0 is None:
        imagePoints = imagePointsList[0]
        modelPoints = modelPointsList[0]
        camera = cameras[0]
        poseParam = cv2.solvePnP(objectPoints = modelPoints, imagePoints = imagePoints[:,None,0:2],\
                                     cameraMatrix = camera_matrix,distCoeffs=None, flags = cv2.SOLVEPNP_EPNP)
        R = cv2.Rodrigues(poseParam[1])[0]
        t = poseParam[2]
        pose = np.hstack((R,t))
        firstCamera = np.hstack((np.eye(3),np.zeros((3,1))))
        pose0 = transformPoseScaled(pose, camera, firstCamera,np.array([1])).squeeze()
    t0 = pose0[:,3]
    R0 = pose0[:,0:3]
    eulAngle0 = rotationMatrixToEulerAngles(R0)
    eulPose0 = np.concatenate((eulAngle0,t0))
    
    sol=scipy.optimize.minimize(lambda eulPose: error(imagePointsList, modelPointsList, eulPose, cameras), eulPose0,tol=0.00000000000001,method='Powell')
    eulImproved = sol.x
    
    RPose = eulerAnglesToRotationMatrix(eulImproved[0:3])
    tPose = eulImproved[3:6]
    improvedPose = np.hstack((RPose,tPose[:,None]))
    return improvedPose













