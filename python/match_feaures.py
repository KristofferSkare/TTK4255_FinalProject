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

I1 = cv.imread('../data_hw5_ext/IMG_8209.jpg', cv.IMREAD_GRAYSCALE)
I2 = cv.imread('../data_hw5_ext/IMG_8211.jpg', cv.IMREAD_GRAYSCALE)

I1_RGB = plt.imread('../data_hw5_ext/IMG_8210.jpg')/255.0
folder = '../data_hw5_ext/calibration'

K  = np.loadtxt(join(folder, 'K.txt'))


def task2_1_a():
    # NB! This script uses a very small number of features so that it runs quickly.
    # You will want to pass other options to SIFT_create. See the documentation:
    # https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
    sift = cv.SIFT_create(nfeatures=30000)
    kp1, desc1 = sift.detectAndCompute(I1, None)
    kp2, desc2 = sift.detectAndCompute(I2, None)
    kp1 = np.array([kp.pt for kp in kp1])
    kp2 = np.array([kp.pt for kp in kp2])

    # NB! You will want to experiment with different options for the ratio test and
    # "unique" (cross-check).
    index_pairs, match_metric = match_features(desc1, desc2, max_ratio=1, unique=True)
    print(index_pairs[:10])
    print('Found %d matches' % index_pairs.shape[0])

    # Plot the 50 best matches in two ways
    best_index_pairs = index_pairs[np.argsort(match_metric)[:50]]
    best_kp1 = kp1[best_index_pairs[:,0]]
    best_kp2 = kp2[best_index_pairs[:,1]]
    plt.figure()
    show_matched_features(I1, I2, best_kp1, best_kp2, method='falsecolor')
    plt.figure()
    show_matched_features(I1, I2, best_kp1, best_kp2, method='montage')
    plt.show()

def estimate_E_SIFT_RANSAC():
     # NB! This script uses a very small number of features so that it runs quickly.
    # You will want to pass other options to SIFT_create. See the documentation:
    # https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
    sift = cv.SIFT_create(nfeatures=30000, sigma=1.6, nOctaveLayers=3, edgeThreshold=12, contrastThreshold=0.04)
    kp1, desc1 = sift.detectAndCompute(I1, None)
    kp2, desc2 = sift.detectAndCompute(I2, None)
    kp1 = np.array([kp.pt for kp in kp1])

    kp2 = np.array([kp.pt for kp in kp2])

    # NB! You will want to experiment with different options for the ratio test and
    # "unique" (cross-check).
    index_pairs, match_metric = match_features(desc1, desc2, max_ratio=0.85, unique=True)
    index_pairs = index_pairs[np.argsort(match_metric)]

    desc1_matched = desc1[index_pairs[:,0]]
    desc2_matched = desc2[index_pairs[:,1]]

    kp1_matched = kp1[index_pairs[:,0]]
    kp2_matched = kp2[index_pairs[:,1]]

    uv1 = np.vstack([kp1_matched.T, np.ones(kp1_matched.shape[0])])
    uv2 = np.vstack([kp2_matched.T, np.ones(kp2_matched.shape[0])])
    xy1 = np.linalg.inv(K)@uv1
    xy2 = np.linalg.inv(K)@uv2

    
    confidence = 0.99
    inlier_fraction = 0.50
    num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)
    distance_threshold = 3.0

    E,inliers = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)
    uv1_in = uv1[:,inliers]
    uv2_in = uv2[:,inliers]
    xy1_in = xy1[:,inliers]
    xy2_in = xy2[:,inliers]
    desc1_in = desc1_matched[inliers, :].T
    desc2_in = desc2_matched[inliers, :].T

    E = estimate_E(xy1_in, xy2_in)

    return E, xy1_in, xy2_in, uv1_in, uv2_in, desc1_in, desc2_in

def task2_1_b():
    E, xy1, xy2, uv1, uv2, desc1_in, desc2_in = estimate_E_SIFT_RANSAC()

    np.random.seed(123) # Comment out to get a random selection each time
    draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)
    plt.show()

def task2_1_d():
    E, xy1, xy2, uv1, uv2, desc1_in, desc2_in = estimate_E_SIFT_RANSAC()
    T4 = decompose_E(E)
    best_num_visible = 0
    for i, T in enumerate(T4):
        P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        P2 = T[:3,:]
        X1 = triangulate_many(xy1, xy2, P1, P2)
        X2 = T@X1
        num_visible = np.sum((X1[2,:] > 0) & (X2[2,:] > 0))
        if num_visible > best_num_visible:
            best_num_visible = num_visible
            best_T = T
            best_X1 = X1
    T = best_T
    X = best_X1
    X = X[:3,:]/X[3,:]
    np.savetxt("3D_points.txt", X.T)
    np.savetxt("descriptors.txt", desc1_in.T)
    draw_point_cloud(X, I1_RGB, uv1, xlim=[-3,+3], ylim=[-3,+1], zlim=[1,5])
    plt.show()

task2_1_d()