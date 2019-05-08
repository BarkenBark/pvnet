from PIL import Image
import numpy as np
from numpy import unique, asarray, zeros, eye,array,swapaxes, stack

resultPath = '/home/andreas/Studies/MasterArbeteLokal/PoseAnnotation/PoseAnnotation/result'
segPath = resultPath + '/Segmentations'

numClasses = 10
identityMatrix = eye(numClasses+1)

img1 = Image.open(segPath + '/seg001.png')
A1=asarray(img1)

img2 = Image.open(segPath + '/seg002.png')
A2=asarray(img2)

imageDim = A1.shape

#instancesInImage = unique(A)

with open(resultPath + "/classindex.txt", "r") as f:
    content = f.readline().rstrip('\n')
    

classIndex = array([0]+[int(x) for x in content.split(',')])

#idxARR=[identityMatrix[x,:] for x in range(numClasses+1)]
A = stack((A1,A2),axis = 2)

imageTensor = identityMatrix[classIndex[A]]
