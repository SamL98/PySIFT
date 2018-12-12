import numpy as np
import numpy.linalg as LA

def get_candidate_keypoints(D):
	candidates = []

	''' Start '''
	# These 2 lines aren't specified in the paper but it makes it so the extrema
	# are found within the entire octave. They are always found in the first or
	# last layer so I probably have something wrong with my DoG pyramid construction.
	D[:,:,0] = 0
	D[:,:,-1] = 0
	''' End '''
	
	for i in range(1, D.shape[0]-1):
		for j in range(1, D.shape[1]-1):
			for k in range(1, D.shape[2]-1): 
				patch = D[i-1:i+2, j-1:j+2, k-1:k+2]
				if np.argmax(patch) == 13 or np.argmin(patch) == 13:
					candidates.append([i, j, k])

	return candidates

def localize_keypoint(D, x, y, s):
	dx = (D[y,x+1,s]-D[y,x-1,s])/2.
	dy = (D[y+1,x,s]-D[y-1,x,s])/2.
	ds = (D[y,x,s+1]-D[y,x,s-1])/2.

	dxx = D[y,x+1,s]-2*D[y,x,s]+D[y,x-1,s]
	dxy = ((D[y+1,x+1,s]-D[y+1,x-1,s]) - (D[y-1,x+1,s]-D[y-1,x-1,s]))/4.
	dxs = ((D[y,x+1,s+1]-D[y,x-1,s+1]) - (D[y,x+1,s-1]-D[y,x-1,s-1]))/4.
	dyy = D[y+1,x,s]-2*D[y,x,s]+D[y-1,x,s]
	dys = ((D[y+1,x,s+1]-D[y-1,x,s+1]) - (D[y+1,x,s-1]-D[y-1,x,s-1]))/4.
	dss = D[y,x,s+1]-2*D[y,x,s]+D[y,x,s-1]

	J = np.array([dx, dy, ds])
	HD = np.array([
		[dxx, dxy, dxs],
		[dxy, dyy, dys],
		[dxs, dys, dss]])
	
	offset = -LA.inv(HD).dot(J)	# I know you're supposed to do something when an offset dimension is >0.5 but I couldn't get anything to work.
	
	return offset, J, HD, x, y, s

def find_keypoints_for_DoG_octave(D, R_th, t_c):
	candidates = get_candidate_keypoints(D)
	#print('%d candidate keypoints found' % len(candidates))

	keypoints = []

	for i, cand in enumerate(candidates):
		y, x, s = cand[0], cand[1], cand[2]
		offset, J, HD, x, y, s = localize_keypoint(D, x, y, s)
		if offset is None:
			continue

		contrast = D[y,x,s] + .5*J.dot(offset)
		if abs(contrast) < t_c: continue

		H = HD[:2, :2]	
		w, v = LA.eig(H)
		r = w[1]/w[0]
		R = (r+1)**2 / r
		if R > R_th: continue

		kp = np.array([x, y, s]) + offset
		keypoints.append(kp)

	#print('%d keypoints found' % len(keypoints))
	return np.array(keypoints)

def get_keypoints(DoG_pyr, R_th, t_c):
    kps = []

    for D in DoG_pyr:
        kps.append(find_keypoints_for_DoG_octave(D, R_th, t_c))

    return kps