# coding: utf-8
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.append(os.pardir)
from dataset.mnist import load_mnist


def show_img(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, normalize=False)

idx = random.randrange(X_train.shape[0])
img = X_train[idx]
label = y_train[idx]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

show_img(img)
