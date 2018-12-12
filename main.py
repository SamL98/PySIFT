from skimage.io import imread
from sift import SIFT

if __name__ == '__main__':
	im = imread('/Users/samlerner/Documents/Misc/IMG_5636.jpeg')

	sift_detector = SIFT(im)
	feats = sift_detector.get_features()

	exit()
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle

	for j in range(len(kps)):
		pts = kps[j]

		_, ax = plt.subplots(1)
		ax.imshow(im, 'gray')
		ax.scatter(pts[:,0], pts[:,1], s=2.5, c='b')

		for pt in pts:
			w, angle = pt[2]*1.5, pt[3]*10
			ax.add_patch(Rectangle(
				(pt[0], pt[1]),
				2*w+1, 2*w+1,
				angle,
				facecolor=None,
				edgecolor='r'))

		ax.set_title('octave: %d' % (j))
		plt.show()

		im = im[::2, ::2]
