#run_utils_deprecated
# Functions that are not actively being used




#### ALTERNATIVE TO AGGREGATION - VIEWING RAY INTERSECTIONS & EPIPOLAR LINES ########

# Function for calculating the epipolar lines on a specific camera belonging to a specific keypoint of all instances in a number of viewpoints
########################################################################################################################################################

def calculate_epipolar_lines(motion, points, projectionIdx=0):
	# motion is an np array of shape [nCameras, 3, 4] containing the camera matrices of each camera
	# points is a list of nCameras tensors with shapes [nInstances, nKeypoints, 2] containing the 2D keypoint projections of all instances in the particular camera
	# or
	# points is a list of nCameras tensors with shapes [nInstances, 2] containing the 2D keypoint projections of all instances in the particular camera
	# projectionIdx is the index of the camera onto which epipolar lines from other cameras shall be projected

	# To be returned
	lines = [] # Should end up with length nCameras, where lines[projectionIdx] = None

	nCameras = motion.shape[0]
	P_proj = motion[projectionIdx] 
	A_proj = P_proj[0:3,0:3]
	for iCamera in range(nCameras):

		# There is no epipolar line of x from the same camera
		if iCamera == projectionIdx:
			lines.append(None)
			continue

		# Find the camera matrices, etc.
		thisPoints = points[iCamera]
		if thisPoints is None:
			lines.append(None)
			continue

		nInstances = points[iCamera].shape[0]
		thisPoints = points[iCamera]
		P_other = motion[iCamera]
		C_other = camera_center(P_other)
		eProj = pflat(P_proj @ pextend(C_other))
		eProjCrossMat = crossmat(eProj.ravel())

		# Calculate the epipolar line of x for each instance of x in this camera
		l = np.zeros((3, nInstances))
		for iInstance in range(nInstances):
			x = pextend(thisPoints[iInstance].reshape(2,1))
			l[:, iInstance] = (eProjCrossMat @ (A_proj @ x)).reshape((3,1)).ravel()
			if -l[0,iInstance]/l[1,iInstance] >= 2:
				print('Epipolar line of center keypoint, instance {}, view {} has high slope.'.format(iCamera, iInstance))

		lines.append(l)

	assert(len(lines)==nCameras)

	return lines



# Function for computing the closest point between two 3D lines
# Based on the formula from morroworks.palitri.com
##################################################################################

def compute_viewing_ray_intersection(ray1, ray2):
	A = ray1[:,0]
	a = ray1[:,1]
	B = ray2[:,0]
	b = ray2[:,1]
	c = B - A # Double check this one
	D = A + a*(-(a@b)*(b@c)+(a@c)*(b@b))/((a@a)*(b@b)-(a@b)*(a@b))
	E = B + b*((a@b)*(a@c)-(b@c)*(a@a))/((a@a)*(b@b)-(a@b)*(a@b))
	return (D+E)/2

# Function for computing the distances between a point x0 and nLines 3D lines
# Based on http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
#####################################################################################

def compute_3d_point_line_distance(x0, lines):
	# X - np array of shape (3,)
	# lines - np array of shape (nLines, 3, 2) where each line is defined by [location (3,), direction (3,)]
	x1 = lines[:,:,0]
	x2 = x1 + lines[:,:,1] 
	distances = np.linalg.norm(np.cross(x0-x1, x0-x2), axis=1)/np.linalg.norm(x2-x1, axis=1)
	return distances # (nLines,)

# Function for computing vieweing rays from each camera in motion to the projection of each instance of a keypoint
# The rays are expressed as 3d lines in a global coordinate system
#########################################################################################################################

def compute_viewing_rays(motion, points):
	viewingRays = np.zeros((0,3,2))
	for iCam, P in enumerate(motion):
		thisPoints = points[iCam] # Instances of the keypoints in this camera, shape (nInstances, 2)
		if thisPoints is not None:
			nInstances = thisPoints.shape[0]
			for iInstance in range(nInstances):
				x = thisPoints[iInstance]
				ray = compute_viewing_ray(P, x)
				viewingRays = np.append(viewingRays, ray[None,:,:], axis=0)

	return viewingRays # np array of shape (nViewingRays, 3, 2) 



# Function for performing Ransac to yield 3D keypoint hypotheses, and inlier counts
##############################################################################################

def ransac_keypoint_3d_hypothesis(motion, points, threshold, nIterations):
	nCameras = len(motion)

	hypotheses = np.zeros((nIterations, 3)) # A hypothesis will be a viewing ray intersection point (representing the 3D location of the specified keypoint)
	inlierCounts = np.zeros((nIterations)) # A viewing ray is an inlier to the hypothesis if its distance to the point is less than threshold

	viewingRays = compute_viewing_rays(motion, points) # np array of shape [nViewingRays, 3, 2] (Note: We no longer care about indexing)

	# Ransac
	for iIteration in range(nIterations):

		sys.stdout.write('\rRansac iteration {} of {}'.format(iIteration+1, nIterations))
		sys.stdout.flush()

		# Sample cameras (viewpoints), make sure each has an instance of the keypoint
		bothHasInstance = False
		while not bothHasInstance:
			cameraIdx = np.random.choice(nCameras, size=(2), replace=False)
			bothHasInstance = (points[cameraIdx[0]] is not None) & (points[cameraIdx[1]] is not None)

		# Sample instance for each viewpoint
		instanceIdx = np.zeros(2, dtype=int)
		for i, iCam in enumerate(cameraIdx):
			nInstances = points[iCam].shape[0]
			instanceIdx[i] = np.random.choice(nInstances)

		# Obtain rays
		x1 = points[cameraIdx[0]][instanceIdx[0]]
		P1 = motion[cameraIdx[0]]
		ray1 = compute_viewing_ray(P1, x1)
		x2 = points[cameraIdx[1]][instanceIdx[1]]
		P2 = motion[cameraIdx[1]]
		ray2 = compute_viewing_ray(P2, x2)

		# Find closest point between rays
		X = compute_viewing_ray_intersection(ray1, ray2) # Global coordinate system

		# Calculate number of inliers
		distances = compute_3d_point_line_distance(X, viewingRays)
		nInliers = np.sum(distances < threshold)

		# Store results
		hypotheses[iIteration, :] = X
		inlierCounts[iIteration] = nInliers


	print('')
	return hypotheses, inlierCounts

# Similar to function above, but let's you sample specific viewpoints, and prints more for debugging
def ransac_keypoint_3d_hypothesis_test(motion, points, threshold, nIterations, cameraIdx, instanceIdx, K):
	nCameras = len(motion)

	hypotheses = np.zeros((nIterations, 3)) # A hypothesis will be a viewing ray intersection point (representing the 3D location of the specified keypoint)
	inlierCounts = np.zeros((nIterations)) # A viewing ray is an inlier to the hypothesis if its distance to the point is less than threshold

	# Ransac
	for iIteration in range(nIterations):

		# Obtain rays
		x1 = points[cameraIdx[0]][instanceIdx[0]]
		P1 = motion[cameraIdx[0]]
		ray1 = compute_viewing_ray(P1, x1)
		x2 = points[cameraIdx[1]][instanceIdx[1]]
		P2 = motion[cameraIdx[1]]
		ray2 = compute_viewing_ray(P2, x2)
		print(' ')
		print('x1: ', K@pextend(x1))
		print('x2: ', K@pextend(x2))
		print('x1 (normalized): ', x1)
		print('x2 (normalized): ', x2)
		print('P1: ', P1)
		print('P2: ', P2)
		print('ray1: ', ray1)
		print('ray2: ', ray2)

		# Find closest point between rays
		X = compute_viewing_ray_intersection(ray1, ray2)
		print(X)

		# Calculate number of inliers
		viewingRays = compute_viewing_rays(motion, points) # np array of shape [nViewingRays, 3, 2] (Note: We no longer care about indexing)
		genRays = np.stack((ray1,ray2))
		generatorDistances = compute_3d_point_line_distance(X, genRays)
		print('genratretre: ', generatorDistances)
		distances = compute_3d_point_line_distance(X, viewingRays)
		nInliers = np.sum(distances < threshold)
		print(' ')
		print(nInliers)
		print(distances)

		# Store results
		hypotheses[iIteration, :] = X
		inlierCounts[iIteration] = nInliers

	return hypotheses, inlierCounts



# Function for deciding the number of 3D centers and their locations, given hypotheses for them
#################################################################################################

def calculate_center_3d(hypothesisPoints, inlierCounts, settings):
	# hypothesisPoints in a (nHypotheses, 3) np array
	# inlierCounts is a (nHypotheses,) np array

	def similarityFun(detection, otherDetections):
		norm = np.linalg.norm((detection - otherDetections), axis=1)
		sim = 1 / norm
		return sim

	similariyThreshold = settings['similariyThreshold']
	neighborThreshold = settings['neighborThreshold']
	scoreThreshold = settings['scoreThreshold']

	# Apply non-maximum supression
	filteredIdx = non_maximum_supression_np_alt(hypothesisPoints, inlierCounts, similariyThreshold, similarityFun, scoreThreshold=scoreThreshold, neighborThreshold=neighborThreshold)
	filteredPoints = hypothesisPoints[filteredIdx,:]
	return filteredPoints
