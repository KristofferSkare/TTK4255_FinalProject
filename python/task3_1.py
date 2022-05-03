from decimal import InvalidContext
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
from scipy.optimize import least_squares
from common import *

from os.path import join






def localize_camera(query):
    model_points = np.loadtxt("3D_points.txt")
    model_desc = np.loadtxt("descriptors.txt", dtype=np.float32)
    folder = '../data_hw5_ext/calibration'
    K  = np.loadtxt(join(folder, 'K.txt'))
    dc = np.loadtxt(join(folder, 'dc.txt'))
    I = cv.imread(query, cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create(nfeatures=30000, sigma=1.6, nOctaveLayers=3, edgeThreshold=12, contrastThreshold=0.04)
    kp, desc = sift.detectAndCompute(I, None)
    kp = np.array([kp_i.pt for kp_i in kp])
    index_pairs, match_metric = match_features(desc, model_desc, max_ratio=0.85, unique=True)
    index_pairs = index_pairs[np.argsort(match_metric)]
    kp_matched = kp[index_pairs[:,0]]
    print(index_pairs.shape)
    #desc_matched = desc[index_pairs[:,0]]
    #model_desc_matched = model_desc[index_pairs[:,1]]
    
    #kp_matched = kp[index_pairs[:,0]]
    model_points_matched = model_points[index_pairs[:,1]]

    print(kp_matched.shape, model_points_matched.shape)
    print(kp_matched[0])
    retval, rvec, tvec, inliers = cv.solvePnPRansac(model_points_matched, kp_matched, K,dc)
    print(retval)
    model_points_matched_inliers = np.array([model_points_matched[i] for i in inliers]) #modify to use inliers
    kp_matched_inliers = np.array([kp_matched[i] for i in inliers])#modify to use inliers
    #print(model_points_matched_inliers[0])

    def resfun(p): 
        u_hat, _ = cv.projectPoints(model_points_matched_inliers, p[:3], p[3:], K, dc)
        vector_errors = (u_hat - kp_matched_inliers)[:,0,:] # the indexing here is because OpenCV likes to add extra dimensions.
        scalar_errors = np.linalg.norm(vector_errors, axis=1)
        print(sum(scalar_errors))
        return scalar_errors

    init = np.array([.5, -.5, 0, 1, -1 ,5])
    #init = np.array([float(rvec[0]),float(rvec[1]), float(rvec[2]), float(tvec[0]), float(tvec[1]), float(tvec[2])]) #transform rvec and tvec into a flat vector for resfun
    ##print(init)
    #print(init[:3])

    #find best paramters
    p = least_squares(resfun, x0=init, method='lm').x
    print(init)
    zero_rot = rotate_x(0)@rotate_y(0)@rotate_z(0)
    T = zero_rot@rotate_x(p[0])@rotate_y([1])@rotate_z(p[2])@translate(p[3], p[4], p[5])
    return T

