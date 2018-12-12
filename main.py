from skimage.io import imread
from sift import SIFT

import pickle

if __name__ == '__main__':
	num_img = 3

	#feats = []
	kp_pyrs = []
	ims = []

	for i in range(1, num_img+1):
		print('Performing SIFT on image %d' % i)

		im = imread('images/IMG_039%d.JPG' % i)
		ims.append(im)

		sift_detector = SIFT(im)
		_ = sift_detector.get_features()
		kp_pyrs.append(sift_detector.kp_pyr)

		pickle.dump(sift_detector.kp_pyr, open('results/kp_pyr%d.pkl' % i, 'wb'))
		pickle.dump(sift_detector.feats, open('results/feat_pyr%d.pkl' % i, 'wb'))

	_, ax = plt.subplots(1, 3)
	ax[0].imshow(ims[0])

	kps = kp_pyrs[0][0]
	ax[0].scatter(kps[:,0], kps[:,1], c='b', s=2.5)

	ax[1].imshow(ims[1])

	kps = kp_pyrs[1][0]
	ax[1].scatter(kps[:,0], kps[:,1], c='b', s=2.5)

	ax[2].imshow(ims[2])

	kps = kp_pyrs[2][0]
	ax[2].scatter(kps[:,0], kps[:,1], c='b', s=2.5)

	plt.show()
