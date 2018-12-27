import numpy as np
from numpy import linalg as LA
from scipy.signal import convolve
from skimage.io import imread
from skimage.transform import warp
import matplotlib.pyplot as plt

from os.path import isfile
import pickle

from sift import SIFT
from match import get_transform, transform_pts

def display_good_keypoints(im1, im2, kps1, kps2, feats1, feats2):
    from match import get_matches
    ix1, ix2 = get_matches(feats1, feats2)
    kps1, kps2 = kps1[ix1], kps2[ix2]

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(im1)
    ax[0].scatter(kps1[:,0], kps1[:,1], c='r', s=3)
    ax[1].imshow(im2)
    ax[1].scatter(kps2[:,0], kps2[:,1], c='b', s=3)
    plt.show()

def gauss_1d_filter(a=0.4):
	return np.expand_dims(
		np.array([.25-.5*a, .25, a, .25, .25-.5*a]),
		axis=1)

def blur_and_sample(img):
	kernel = gauss_1d_filter()
	img = convolve(img, kernel)
	img = convolve(img, kernel.T)
	return img[::2, ::2]

def interpolate(img):
	new = np.zeros((2*img.shape[0]-1, 2*img.shape[1]-1))
	new[::2, ::2] = img

	col_avg = (img[:,1:] + img[:,:-1])/2
	row_avg = (img[1:,:] + img[:-1,:])/2
	new[::2, 1::2] = col_avg
	new[1::2, ::2] = row_avg

	avg_sum = (row_avg + col_avg.T)
	new[1::2, 1::2] = (avg_sum[:,1:] + avg_sum[:,:-1])/4	
	return new

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
    return detector.kp_pyr[0], detector.feats[0]

def corners(h, w):
    return np.array([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

def get_transformed_corners(H, h, w):
    return transform_pts(corners(h, w), H)

img_dir = 'panorama'
feats_dir = 'panorama_feats'

num_img = 2
start_ix = 3

prev_im = (imread(img_dir+'/IMG_040%d.JPG' % (start_ix+1))/255.).astype(np.float64)
kps_prev, feats_prev = extract_or_load_features(prev_im, feats_dir+'/kp_pyr%d.pkl'%start_ix, feats_dir+'/feat_pyr%d.pkl'%start_ix)

i = start_ix+1
im = (imread(img_dir+'/IMG_040%d.JPG' % (i+1))/255.).astype(np.float64)
kps, feats = extract_or_load_features(im, feats_dir+'/kp_pyr%d.pkl'%i, feats_dir+'/feat_pyr%d.pkl'%i)
H = get_transform(prev_im, im, kps_prev, kps, feats_prev, feats, ratio=.8)

display_good_keypoints(prev_im, im, kps_prev, kps, feats_prev, feats)

transformed_corners = get_transformed_corners(H, *(im.shape[:-1]))
corner_offset = corners(*(im.shape[:-1])) - np.abs(transformed_corners)

h, w , _= prev_im.shape

min_ty = int(transformed_corners[:,1].min())
max_ty = int(transformed_corners[:,1].max())

if min_ty < 0: h += abs(min_ty)
if max_ty > h: h += max_ty-h

min_tx = int(transformed_corners[:,0].min())
max_tx = int(transformed_corners[:,0].max())

if min_tx < 0: w += abs(min_tx)
if max_tx > w: w += max_tx-w

canvas = np.zeros((h, w, 3), dtype=np.float64)

t = abs(min_ty)
l = abs(min_tx)
canvas[t:t+prev_im.shape[0], l:l+prev_im.shape[1]] = prev_im.copy()
canvas = warp(canvas, LA.inv(H))

t = max(0, -min_ty)
l = max(0, -min_tx)
canvas[t:t+im.shape[0], l:l+im.shape[1]] = im.copy()

plt.imshow(canvas)
plt.show()
