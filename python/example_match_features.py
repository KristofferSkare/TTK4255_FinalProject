from operator import index
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matlab_inspired_interface import match_features, show_matched_features
import matplotlib.pyplot as plt
import numpy as np
from hw5_sol.figures import *
from hw5_sol.estimate_E import *
from hw5_sol.decompose_E import *
from hw5_sol.triangulate_many import *
from hw5_sol.epipolar_distance import *
from hw5_sol.estimate_E_ransac import *
from hw5_sol.F_from_E import *
#from draw_point_cloud import draw_point_cloud

I1 = cv.imread('../data_hw5_ext/IMG_8210.jpg', cv.IMREAD_GRAYSCALE)
I2 = cv.imread('../data_hw5_ext/IMG_8211.jpg', cv.IMREAD_GRAYSCALE)

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
index_pairs, match_metric = match_features(desc1, desc2, max_ratio=0.9, unique=False)
print(index_pairs[:10])
print('Found %d matches' % index_pairs.shape[0])


#egen kode
matches = np.array([[kp1[pair[0]][0],kp1[pair[0]][1], kp2[pair[1]][0], kp2[pair[1]][1]] for pair in index_pairs])
print(matches[0])
print(matches.shape[0])
K = np.loadtxt('../data_hw5_ext/calibration/K.txt')
ransac = True
########################################################
#direct import from hw5
uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])
xy1 = np.linalg.inv(K)@uv1
xy2 = np.linalg.inv(K)@uv2

if ransac:
    e = epipolar_distance(F_from_E(np.loadtxt('hw5_sol\E.txt'), K), uv1, uv2)
    plt.figure('Histogram of epipolar distances')
    plt.hist(np.absolute(e), range=[0, 40], bins=100, cumulative=True)
    plt.title('Cumulative histogram of |epipolar distance| using good E')
    plt.xlabel('Absolute epipolar distance (pixels)')
    plt.ylabel('Occurrences')

    distance_threshold = 4.0

    # Automatically computed trial count
    confidence = 0.99
    inlier_fraction = 0.50
    num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)

    # Alternatively, hard-coded trial count
    num_trials = 2000

    E,inliers = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)
    uv1 = uv1[:,inliers]
    uv2 = uv2[:,inliers]
    xy1 = xy1[:,inliers]
    xy2 = xy2[:,inliers]

E = estimate_E(xy1, xy2)

if not ransac:
    np.savetxt('E.txt', E)

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
print('Best solution: %d/%d points visible' % (best_num_visible, xy1.shape[1]))
print(I1)
np.random.seed(123) # Comment out to get a random selection each time
#draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)
draw_point_cloud(X, I1, uv1,  xlim=[-5,+5], ylim=[-5,+5], zlim=[1,15])
plt.show()
#end of direct import hw5
###############################################################################

# Plot the 50 best matches in two ways
best_index_pairs = index_pairs[np.argsort(match_metric)[:50]]
best_kp1 = kp1[best_index_pairs[:,0]]
best_kp2 = kp2[best_index_pairs[:,1]]
plt.figure()
show_matched_features(I1, I2, best_kp1, best_kp2, method='falsecolor')
plt.figure()
show_matched_features(I1, I2, best_kp1, best_kp2, method='montage')
plt.show()
