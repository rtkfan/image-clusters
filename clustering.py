from icecream import ic
import os

import numpy as np

DATADIR = './images'

with os.scandir(DATADIR) as files:
    filenames = [ifile.name for ifile in files if ifile.name.endswith(('.png','.jpg','.gif'))]
