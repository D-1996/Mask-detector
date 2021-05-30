# -*- coding: utf-8 -*-
import cv2
import os
import pickle
import numpy as np


ROOT = 'C:/Users/damia/OneDrive/Pulpit/Git/FM detector/'
DATA_LOC = ROOT + 'data/'
CLASSES = ['with_mask', 'without_mask']
IMG_SIZE = 120


def create_dataset_from_img(size):
    X = []
    y = []
    
    for cl in CLASSES:
        path = os.path.join(DATA_LOC, cl)
        class_num = CLASSES.index(cl)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (size, size))
                new_array = np.transpose(new_array, (2, 0, 1)).astype(np.float32)
                X.append(new_array)
                y.append(class_num)
            except Exception as e:
                print('{} problem with image {}'.format(e, os.path.join(path,img)))

    return X, y


features, targets = create_dataset_from_img(IMG_SIZE)


with open(os.path.join(ROOT, 'pickle/features.Pickle'), 'wb') as f:
    pickle.dump(features, f)
    
with open(os.path.join(ROOT, 'pickle/targets.Pickle'), 'wb') as f:
    pickle.dump(targets, f)

assert len(features) == len(targets)