import numpy as np

def non_maximum_supression(detections, scores, threshold, similarityFun):
	# Input: 
	# detections - list of data point candidates detected by algorithm, e.g. bounding boxes for object detection
	# scores - list of scores associated with the detections
	# threshold - similarity threshold, discard detections too similar to the currently top-scoring detection
	# similarityFun - function which returns a list of similarity measures between current detection (arg1) and other detections in a list (arg2)

	# Output:
	# filteredDetectionIdx - list of indices of 'detections' that remain after NMS

	assert(len(scores) == len(detections))

	remainingIdx = scores.argsort()[::-1] # Scores for detections in descending order
	filteredDetectionIdx = [] # To be filled with the filtered detection indices

	while remainingIdx.size() > 0:
		# Determine remaining detections
		idx = remainingIdx[0]
		filteredDetectionIdx.append(idx)
		thisDetection = detections[idx]
		remainingDetections = detections[remainingIdx[1:]]

		# Calculate the similiarity between thisDetection and all other remaining detections
		sim = similarityFun(thisDetection, remainingDetections)

		# Remove detections too similar to thisDetection
		keepIdx = np.where(sim <= threshold)
		remainingIdx = remainingIdx[keepIdx + 1]

def non_maximum_supression_np(detections, scores, similarityThreshold, similarityFun, scoreThreshold=None, neighborThreshold=None):
	# Input: 
	# detections - (1+N)D-np.ndarray of data point candidates detected by algorithm, e.g. bounding boxes for object detection. First dimension is the batch dimension.
	# scores - 1D-np.ndarray of scores associated with the detections. 
	# threshold - similarity threshold, discard detections too similar to the currently top-scoring detection
	# similarityFun - function which returns an np.ndarray of similarity measures between current detection (arg1) and other detections (arg2)

	# Output:
	# filteredDetectionIdx - list of indices of 'detections' that remain after NMS

	assert(scores.shape[0] == detections.shape[0])

	remainingIdx = scores.argsort()[::-1] # Scores for detections in descending order
	filteredDetectionIdx = [] # To be filled with the filtered detection indices

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
				filteredDetectionIdx.append(idx)
		else:
			filteredDetectionIdx.append(idx)

	return filteredDetectionIdx