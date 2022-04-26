import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matlab_inspired_interface import match_features, show_matched_features
from hw5.estimate_E_ransac import estimate_E_ransac,get_num_ransac_trials
from hw5.F_from_E import F_from_E
from hw5.figures import draw_correspondences, draw_point_cloud
from hw5.estimate_E import estimate_E
from hw5.decompose_E import decompose_E
from hw5.triangulate_many import triangulate_many

from os.path import join

I = cv.imread('../data_hw5_ext/IMG_8209.jpg', cv.IMREAD_GRAYSCALE)
model_points = np.loadtxt("3D_points.txt")
model_desc = np.loadtxt("descriptors.txt", dtype=np.float32)

folder = '../data_hw5_ext/calibration'

K  = np.loadtxt(join(folder, 'K.txt'))


def localize_camera():

    sift = cv.SIFT_create(nfeatures=30000, sigma=1.6, nOctaveLayers=3, edgeThreshold=12, contrastThreshold=0.04)
    kp, desc = sift.detectAndCompute(I, None)
    kp = np.array([kp_i.pt for kp_i in kp])

    index_pairs, match_metric = match_features(desc, model_desc, max_ratio=0.85, unique=True)
    index_pairs = index_pairs[np.argsort(match_metric)]
    print(index_pairs.shape)
    desc_matched = desc[index_pairs[:,0]]
    model_desc_matched = model_desc[index_pairs[:,1]]
    
    kp_matched = kp[index_pairs[:,0]]
    model_points_matched = model_points[index_pairs[:,1]]

    print(kp_matched.shape, model_points_matched.shape)
  

localize_camera()