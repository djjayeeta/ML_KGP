import numpy as np
import scipy.io
from scipy import sparse
from sklearn import preprocessing
import hybrid_grad_helper as hgh
from numpy import linalg as LA
from helper import image_helper as ih
import pickle

def threshold_u(u):
	r, cluster_no = u.shape
	index = np.argmax(u,axis=1)
	uhard = np.zeros((r, cluster_no))
	for i in xrange(0,r):
		uhard[i][index[i]] = 1
	return uhard


def assing_classes(U,m,n):
	L = [[0 for i in xrange(n)] for j in xrange(m)]
	for j in xrange(0,n):
		for i in xrange(0,m):
			L[i][j] = U[j*m+i].argmax()
	return L


def pd_nonlocal_HSI_stop_new(image,W,mu,endmem,lamda,tao,sigma,theta,tol,iter_stop,innerloop,outerloop,output_path):
	image = image.astype(dtype=np.float64)
	m,n,channel = image.shape
	channel,cluster_no = endmem.shape
	uhard_old = np.zeros((m*n, cluster_no))
	stop = 0
	r = m*n
	u = np.ones((r,cluster_no))/cluster_no
	u_bar = np.array(u, copy=True)
	p = [sparse.csc_matrix((r,r),dtype=np.float64) for i in xrange(0,cluster_no)]
	count = 0
	diff = 1
	outer_index = 0
	error = np.zeros((innerloop*outerloop,1))
	image_2d = np.reshape(image, (r,channel),order='F')
	image_2d_normalized =  preprocessing.normalize(image_2d,norm='l2', axis=1)
	W_sqrt = W
	W.data **= 0.5

	while stop < iter_stop and count < innerloop*outerloop:
		outer_index += 1
		endmem_normalized = preprocessing.normalize(endmem,norm='l2', axis=0)
		temp = np.ones((r,cluster_no)) - np.matmul(image_2d_normalized,endmem_normalized) + mu * hgh.eu_distance(image_2d, endmem.transpose())**0.5
		f = 0.5*lamda*temp**2
		
		for jj in xrange(0,innerloop):
			count += 1
			uold = np.array(u, copy=True)
			
			for l in xrange(0,cluster_no):
				p[l] = p[l] + hgh.nonlocal_grad_new(W_sqrt, sigma*u_bar[:,l])
				tempp = sparse.csr_matrix(p[l], copy=False)
				tempp.data **= 2
				coe = np.array((tempp.sum(axis=1)))**0.5
				coe = np.amax(coe,axis=1)
				coe[np.where(coe<1.0)] = 1
				diagcoe = sparse.lil_matrix((r,r),dtype=np.float64)
				diagcoe.setdiag(1/coe)
				p[l] = diagcoe*p[l]
				u[:,l] = u[:,l] + hgh.nonlocal_divergence_new(W_sqrt, tao*p[l]) - tao*f[:,l]
			u = hgh.projsplx_multi(u)
			diff = LA.norm(u-uold,'fro')/LA.norm(u,'fro')
			u_bar = u+theta*(u-uold)
			error[count] = diff

		uhard = threshold_u(u)
		for l in xrange(0,cluster_no):
			index = np.nonzero(uhard[:,l])
			num_pixel = len(index[0])
			endmem[:,l] = (np.sum(image_2d[index],axis=0)/num_pixel)
		L = assing_classes(uhard,m,n)
		ih.save_image(L,output_path + "_" + str(outer_index) + ".jpeg")
		iter_sparse = uhard_old-uhard
		stop = 1- (np.nonzero(iter_sparse)[0].shape[0])/r
		uhard_old = uhard
	return error

def run_my_func():
	mat_file_path = "data_urban.mat"
	mat = scipy.io.loadmat(mat_file_path)
	f = mat['H']
	W = mat['W_binary_10']
	mu = 1e-5
	lamda = 1e6
	tao = 10.0
	sigma = 1.0/tao/4.0/10.0
	endmem_rand = mat['endmem_rand']
	sigma = sigma/3
	theta = 1
	tol = 1e-2
	iter_stop = 1.1
	innerloop = 5
	outerloop = 50
	output_path = "data/linear_hybrid_output_"
	endmem = endmem_rand
	image = f
	error = pd_nonlocal_HSI_stop_new(f,W,mu,endmem_rand,lamda,tao,sigma,theta,tol,iter_stop,innerloop,outerloop,output_path)
	
