# Test get_ms_bw_limits

import os
import sys
import time
import pickle
sys.path.append('.')
sys.path.append('..')

from lib.ica.utils import parse_3D_keypoints
from lib.ica.run_utils import get_ms_bandwidth_limits

modelDir = '/var/www/webdav/Data/ICA/Models/models_meter'
modelIdx = 10
lower, upper = get_ms_bandwidth_limits(modelDir, modelIdx)
print(lower, upper)