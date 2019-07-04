# Compare poses'
# Import statements
import os
import sys
sys.path.append('.')
sys.path.append('..')

# Other modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
import itertools


# Own modules
from lib.ica.utils import *
from lib.ica.run_utils import covar_to_weight

# PVNet modules
from lib.utils.data_utils import read_rgb_np
from lib.utils.evaluation_utils import pnp
from lib.utils.extend_utils.extend_utils import uncertainty_pnp





#################################################################################################
#################################################################################################
#################################################################################################





posesEstPath = '/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10/poses_from_gt.pickle'
posesGtPath= '/var/www/webdav/Data/ICA/Results/Experiment4/Scene13/Class10/poses_gt.pickle'

posesEst = pickleLoad(posesEstPath)
posesGt = pickleLoad(posesGtPath)

print(len(posesEst))
print(len(posesGt))

for i in range(253):
	print('------------------------')
	print('posesEst:')
	print(posesEst[i])
	print()
	print('posesGt:')
	print(posesGt[i])
	print('------------------------')