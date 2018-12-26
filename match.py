import numpy as np
from numpy import linalg as LA

from skimage.io import imread
from skimage.transform import warp
import matplotlib.pyplot as plt

import pickle

ims = []
kp_pyrs = []
feat_pyrs = []

for i in range(3):
	ims.append(imread('images/IMG_039%d.JPG' % (i+1)))
	kp_pyrs.append(pickle.load(open('results/kp_pyr%d.pkl' % (i+1), 'rb')))
	feat_pyrs.append(pickle.load(open('results/feat_pyr%d.pkl' % (i+1), 'rb')))

def compute_homography(pts1, pts2):
	pts1 = np.concatenate((pts1, np.ones((len(pts1), 1))), axis=1)
	pts2 = np.concatenate((pts2, np.ones((len(pts2), 1))), axis=1)
	
	A = np.zeros((2*len(pts1), 9))
	for i in range(0, 2*len(pts1), 2):
		ptTo = pts2[i//2]
		ptFrom = pts1[i//2]

		A[i][:3] = ptFrom
		A[i+1][3:6] = ptFrom
		A[i][6:] = -ptFrom * ptTo[0]
		A[i+1][6:] = -ptFrom * ptTo[1]

	eigvals, eigvecs = LA.eig(A.T.dot(A))
	H = eigvecs[:,np.argmin(eigvals)].reshape((3, 3))

	if LA.norm(H) != 1.:
		H /= LA.norm(H)

	return H

def get_matches(feats1, feats2):
	idxs1, idxs2 = [], []
	for i, feat in enumerate(feats1):
		distances = LA.norm(feats2-feat, axis=1)
		nn = np.argsort(distances)[:2]
		dist1, dist2 = distances[nn[0]], distances[nn[1]]

		if dist1/max(1e-6, dist2) < .9:
			idxs1.append(i)
			idxs2.append(nn[0])

	return idxs1, idxs2

def find_good_homography(kps1, kps2, n_trials=500):
	best_H = None
	max_inliers = -1

	def calculate_num_inliers(H, pts1, pts2):
		pts1 = np.concatenate((pts1, np.ones((len(pts1), 1))), axis=1)
		pts2_hat = pts1.dot(H.T)
		pts2_hat = np.array([pts2_hat[:,0]/pts2_hat[:,2], pts2_hat[:,1]/pts2_hat[:,2]]).T
		distances = LA.norm(pts2-pts2_hat, axis=1)
		return np.sum(distances<5)

	for _ in range(n_trials):
		chosen_idxs = np.random.choice(range(len(kps1)), 4, replace=False)
		H = compute_homography(kps1[chosen_idxs], kps2[chosen_idxs])

		num_inliers = calculate_num_inliers(H, kps1, kps2)
		if num_inliers > max_inliers:
			max_inliers = num_inliers
			best_H = H

	return best_H


for i in range(3):
	for j in range(i+1, 3):
		im1, im2 = ims[i], ims[j]
		feats1, feats2 = feat_pyrs[i][0], feat_pyrs[j][0]
		kps1, kps2 = kp_pyrs[i][0], kp_pyrs[j][0]

		idxs1, idxs2 = get_matches(feats1, feats2)
		kps1, kps2 = kps1[idxs1,:-2], kps2[idxs2,:-2]
		H = find_good_homography(kps1, kps2)

		_, ax = plt.subplots(2, 2)
		ax[0,0].imshow(im1)
		ax[0,0].axis('off')

		ax[0,1].imshow(im2) 
		ax[0,1].axis('off')

		ax[1,0].imshow(warp(im1, LA.inv(H)))
		ax[1,0].axis('off')

		ax[1,1].imshow(warp(im2, H))
		ax[1,1].axis('off')

		plt.show()