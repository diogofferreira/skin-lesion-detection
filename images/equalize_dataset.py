import cv2
import numpy as np
from skimage import data, img_as_float
from skimage import exposure
from os import listdir
from os.path import isfile, join
from functools import reduce

imgs = ['1_cropped.jpg', '2_cropped.jpg', '3_cropped.jpg']

for f in imgs:
    im = cv2.imread(f)
    p2, p98 = np.percentile(im, (2, 98)) #####
    im = exposure.rescale_intensity(im, in_range=(p2, p98))

    cv2.imwrite(f[0] + '_hist.jpg', im)

