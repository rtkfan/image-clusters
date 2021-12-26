from icecream import ic
import os
import logging

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
NUM_CLUSTERS = 3

with os.scandir(DATADIR) as files:
    filenames = [ifile.name for ifile in files
                   if ifile.name.endswith(('.png','.jpg','.gif'))]

logging.info(f'Found {len(filenames)} images in data directory')

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

data = {}

for ifile in filenames[:5]:
    load_path = os.path.join(DATADIR,ifile)
    logging.info(f'Load {load_path}')
    feat = extract_features(load_path,model)
    data[ifile] = feat

feat = np.array(list(data.values()))
feat = feat.reshape(-1,4096)

pca = PCA(n_components=3, random_state = 22)
pca.fit(feat)
x = pca.transform(feat)

kmeans = KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(x)

ic(kmeans.labels_)
