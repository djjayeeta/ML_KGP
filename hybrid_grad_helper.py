import numpy as np
from scipy import sparse

def nonlocal_grad_new(W_sqrt, u):
	r = u.shape[0]
	diagu = sparse.lil_matrix((r,r),dtype=np.float64)
	diagu.setdiag(u)
	p1 = W_sqrt * diagu
	p2 = diagu * W_sqrt
	p = p1 - p2
	return p


def nonlocal_divergence_new(W_sqrt,v ):
	temp = W_sqrt.multiply(v)
	temp = temp - temp.transpose()
	div = temp.sum(axis=1)
	return np.array(div)[:,0]

def eu_distance(x, c):
	r,channel = x.shape
	cluster_no, channel = c.shape
	tempx = (x**2).sum(axis=1)
	tempc = (c**2).sum(axis=1)
	tempxc = np.matmul(x,c.transpose())
	tempx1 = np.repeat((tempx * np.ones((1,1))).transpose(),cluster_no,axis=1)
	tempc2 = np.repeat(np.ones((1,1))*tempc,r,axis=0)
	dist = tempx1 + tempc2 - 2*tempxc
	return dist



def projsplx_multi(Y):
	n, m = Y.shape
	if n==1:
	    X = projsplx(Y)
	else:
		Y1 = -np.sort(-Y,axis=1)
		tmpsum = np.zeros(n)
		tmax = np.zeros(n)
		bget = np.zeros(n, dtype=bool)
		for ii in xrange(0,m-1):
			active = np.where(bget==False)
			tmpsum[active] = tmpsum[active] + Y1[active][:,ii]
			tmax[active] = (tmpsum[active] - 1)/(ii+1)
			deactivate = np.intersect1d(np.nonzero(tmax>=Y1[:,ii+1]), active)
			bget[deactivate] = True
		active = np.where(bget==False)
		tmax[active] = (tmpsum[active] + Y1[active][:,m-1] - 1)/m
		X = (Y.transpose() - tmax).transpose()
		X[np.where(X<0.0)] = 0.0
	return X 


def projsplx(y):
	y1 = np.array(y, copy=True)
	m = y1.shape[1]
	bget = False
	y1[0][::-1].sort()
	tmpsum = 0
	for ii in xrange(0,m-1):
		tmpsum = tmpsum + y1[0][ii]
		tmax = (tmpsum - 1)/ii
		if tmax >= y1[0][ii+1]:
			bget = True
			break
	if not bget:
		tmax = (tmpsum + y1[0][m] -1)/m
	y1 = y1 - tmax
	y1[np.where(y1<0.0)] = 0.0
	return y1


