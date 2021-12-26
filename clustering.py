from icecream import ic
import os
import logging

import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

DATADIR = './images'

with os.scandir(DATADIR) as files:
    filenames = [ifile.name for ifile in files
                   if ifile.name.endswith(('.png','.jpg','.gif'))]

logging.info(f'Found {len(filenames)} images in data directory')
