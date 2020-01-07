import numpy as np
from numpy import linalg as LA

from skimage.io import imread
from skimage.transform import warp
import matplotlib.pyplot as plt

import pickle
from os.path import isfile

from sift import SIFT

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

def get_matches(feats1, feats2, ratio=0.8):
	idxs1, idxs2 = [], []
	for i, feat in enumerate(feats1):
		distances = LA.norm(feats2-feat, axis=1)
		nn = np.argsort(distances)[:2]
		dist1, dist2 = distances[nn[0]], distances[nn[1]]

		if dist1/max(1e-6, dist2) < ratio:
			idxs1.append(i)
			idxs2.append(nn[0])

	return idxs1, idxs2

def transform_pts(pts, H):
	pts = np.concatenate((pts, np.ones((len(pts), 1))), axis=1)
	trans = pts.dot(H.T)
	return np.array([trans[:,0]/trans[:,2], trans[:,1]/trans[:,2]]).T

def find_good_homography(kps1, kps2, n_trials=500):
	best_H = None
	max_inliers = -1

	def calculate_num_inliers(H, pts1, pts2):
		pts2_hat = transform_pts(pts1, H)
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

def get_transform(im1, im2, kps1, kps2, feats1, feats2, ratio=0.8, ret_idxs=False):
	idxs1, idxs2 = get_matches(feats1, feats2, ratio=ratio)
	kps1, kps2 = kps1[idxs1,:-2], kps2[idxs2,:-2]
	if ret_idxs:
		return find_good_homography(kps1, kps2), idxs1, idxs2
	return find_good_homography(kps1, kps2)

def extract_or_load_features(im, kp_fname, feat_fname):
    if isfile(kp_fname) and isfile(feat_fname):
        #return pickle.load(open(kp_fname, 'rb'))[0], pickle.load(open(feat_fname, 'rb'))[0]
        kps, feats = pickle.load(open(kp_fname, 'rb')), pickle.load(open(feat_fname, 'rb'))
        kps = np.concatenate(kps, axis=0)
        feats = np.concatenate(feats, axis=0)
        return kps, feats

    detector = SIFT(im)
    _ = detector.get_features()
    pickle.dump(detector.kp_pyr, open(kp_fname, 'wb'))
    pickle.dump(detector.feats, open(feat_fname, 'wb'))
    return np.concatenate(detector.kp_pyr, axis=0), np.concatenate(detector.feats, axis=0)

if __name__ == '__main__':
	ims = []
	kp_pyrs = []
	feat_pyrs = []

	i = 1
	j = i+1
	im_dir = 'images/'
	im_fmt = 'IMG_039%d.JPG'
	feat_dir= 'results'

	for ix in [i, j]:
		ims.append(imread(im_dir+im_fmt % ix))
		kps, feats = extract_or_load_features(ims[-1], feat_dir+'/kp_pyr%d.pkl' % ix, feat_dir+'/feat_pyr%d.pkl' % ix)
		kp_pyrs.append(kps)
		feat_pyrs.append(feats)

	im1, im2 = ims[0], ims[1]
	H, ix1, ix2 = get_transform(im1, im2, kp_pyrs[0], kp_pyrs[1], feat_pyrs[0], feat_pyrs[1], ret_idxs=True)
	kps1 = kp_pyrs[0][ix1]
	kps2 = kp_pyrs[1][ix2]

	_, ax = plt.subplots(2, 2)
	ax[0,0].imshow(im1)
	ax[0,0].scatter(kps1[:,0], kps1[:,1], c='r', s=3)
	ax[0,0].axis('off')

	ax[1,0].imshow(im2) 
	ax[1,0].scatter(kps2[:,0], kps2[:,1], c='r', s=3)
	ax[1,0].axis('off')

	ax[1,1].imshow(warp(im1, LA.inv(H)))
	ax[1,1].scatter(kps2[:,0], kps2[:,1], c='r', s=3)
	ax[1,1].axis('off')

	ax[0,1].imshow(warp(im2, H))
	ax[0,1].scatter(kps1[:,0], kps1[:,1], c='r', s=3)
	ax[0,1].axis('off')

	plt.show()
