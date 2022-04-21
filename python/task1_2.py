import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import cv2 as cv


folder = '../data_hw5_ext/calibration'
image_folder = "../data_hw5_ext/"
image = "IMG_8207.jpg"
image_path = image_folder  + image 

K           = np.loadtxt(join(folder, 'K.txt'))
dc          = np.loadtxt(join(folder, 'dc.txt'))
std_int     = np.loadtxt(join(folder, 'std_int.txt'))

fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

dc_std = np.array([k1, k2, p1, p2, k3])
N = 9

I = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
images = []
for i in range(N):
    coeff = np.random.normal(dc, dc_std)
    
    out = cv.undistort(I, K, coeff, )
    images.append(out)

mean_undistort = cv.undistort(I, K, dc)

plt.figure(1)
for index, i in enumerate(images):
    plt.subplot(330 + index+1)
    plt.imshow(i, cmap="gray")

plt.figure(2)
for index, i in enumerate(images):
    plt.subplot(330 + index+1)
    plt.imshow(np.abs(mean_undistort - i), cmap="gray")

plt.show()



