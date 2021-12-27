from icecream import ic
import os
import logging
from tqdm import tqdm

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import numpy as np


def extract_features(file, model):
    # load image and make it a numpy array
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)

    # append dimension for number of images
    reshaped_img = img.reshape(1,224,224,3)

    # preprocess image and get feature vector
    img_prepro = preprocess_input(reshaped_img)
    features = model.predict(img_prepro, use_multiprocessing=True)

    return features


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

DATADIR = './images'

with os.scandir(DATADIR) as files:
    filenames = [ifile.name for ifile in files
                   if ifile.name.endswith(('.png','.jpg','.gif'))]

logging.info(f'Found {len(filenames)} images in data directory')

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

data = {}

logging.info(f'Loading and preprocessing {len(filenames)} images from datadir')
for ifile in tqdm(filenames):
    load_path = os.path.join(DATADIR,ifile)
    feat = extract_features(load_path,model)
    data[ifile] = feat

feat = np.array(list(data.values()))
feat = feat.reshape(-1,4096)

pca = PCA(n_components=0.95)
pca.fit(feat)
x = pca.transform(feat)

logging.info(f'PCA reduced dimensionality to {pca.n_components_}')

i = 1
objective = 1

while i <= len(filenames) and objective > 1e-9:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    objective = kmeans.inertia_
    testout = max(kmeans.labels_)
    logging.info(f'K-Means with {i} clusters yields inertia {objective}')
    i+=1
