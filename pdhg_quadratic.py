import numpy as np
import scipy.io
from scipy import sparse
from sklearn import preprocessing
from numpy import linalg as LA
from helper import image_helper as ih
import pickle,timeit
import numpy.matlib
from multiprocessing import Pool
from stable_simplex import threshold_u
maxconn = 8

def nonlocal_grad(W_sqrt, u):
	r = u.shape[0]
	diagu = sparse.lil_matrix((r,r),dtype=np.float64)
	diagu.setdiag(u)
	p1 = W_sqrt * diagu
	p2 = diagu * W_sqrt
	p = p1 - p2
	return p


def nonlocal_divergence(W_sqrt,v ):
	temp = W_sqrt.multiply(v)
	temp = temp - temp.transpose()
	div = temp.sum(axis=1)
	return np.array(div)[:,0]

def eu_distance(x, c):
	r,channel = x.shape
	cluster_no, channel = c.shape
	tempxc = np.matmul(x,c.transpose())
	tempx1 = np.repeat((((x**2).sum(axis=1)) * np.ones((1,1))).transpose(),cluster_no,axis=1)
	tempc2 = np.repeat(np.ones((1,1))*((c**2).sum(axis=1)),r,axis=0)
	dist = tempx1 + tempc2 - 2*tempxc
	return dist

def cond_projsplx_multi(Y,a_mat):
	n, m = Y.shape
	A = a_mat
	s = -np.sort(-(A*Y),axis=1)
	index = np.argsort(-(A*Y), axis=1)
	tmpsum = np.zeros(n) 
	tmpsumdom = np.zeros(n)
	bget = np.zeros(n, dtype=bool)
	A_sort = A[np.arange(np.shape(A)[0])[:,np.newaxis], index]
	cond_s = s/(A_sort**2)
	tmax = np.zeros(n)
	for ii in xrange(0,m-1):
		active = (bget==False)
		tmpsum[active] = tmpsum[active] + cond_s[active][:,ii]
		tmpsumdom[active] = tmpsumdom[active]+ 1.0/A_sort[active][:,ii]**2
		tmax[active] = (tmpsum[active] - 1)/tmpsumdom[active]
		deactivate = (tmax >= s[:,ii+1]) & active
		bget[deactivate] = True
	active = (bget==False)
	tmax[active] = (tmpsum[active] + cond_s[active][:,m-1] - 1)/(tmpsumdom[active]+1.0/(A_sort[active][:,m-1])**2)
	X = (Y - np.matlib.repmat(tmax.reshape(n,1),1,m)*1.0/A)
	X[X<0.0] = 0.0
	X = X/A
	return X

# def threshold_u(u):
# 	r, cluster_no = u.shape
# 	index = np.argmax(u,axis=1)
# 	uhard = np.zeros((r, cluster_no))
# 	for i in xrange(0,r):
# 		uhard[i][index[i]] = 1
# 	return uhard

def project_p(p,r):
	tempp = sparse.csr_matrix(p, copy=False)
	tempp.data **= 2
	coe = np.array((tempp.sum(axis=1)))**0.5
	coe = np.amax(coe,axis=1)
	coe[coe<1.0] = 1
	diagcoe = sparse.lil_matrix((r,r),dtype=np.float64)
	diagcoe.setdiag(1/coe)
	p = diagcoe*p
	return p


def assing_classes(U,m,n):
	L = [[0 for i in xrange(n)] for j in xrange(m)]
	for j in xrange(0,n):
		for i in xrange(0,m):
			L[i][j] = U[j*m+i].argmax()
	return L

def calculate_centroid(uhard,cluster_no,endmem,image_2d):
	for l in xrange(0,cluster_no):
		index = np.nonzero(uhard[:,l])
		num_pixel = len(index[0])
		endmem[:,l] = (np.sum(image_2d[index],axis=0)/num_pixel)
	return endmem

def get_stop(uhard_old,uhard,r):
	iter_sparse = uhard_old - uhard
	stop = 1- (np.nonzero(iter_sparse)[0].shape[0])/r
	return stop

def calculate_cluster_wise_pdhg(params):
	u_bar_lth,sigma,tao,W_sqrt,l,r,p_lth,f_lth,u_lth = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
	p_lth = p_lth + nonlocal_grad(W_sqrt, sigma*u_bar_lth)
	p_lth = project_p(p_lth,r)
	u_lth = u_lth + nonlocal_divergence(W_sqrt, tao*p_lth) 
	u_lth = (1.0/(tao*f_lth+1)**0.5)*u_lth
	return [p_lth,u_lth]


def pd_nonlocal_HSI(image,W,mu,endmem,lamda,tao,sigma,theta,tol,iter_stop,innerloop,outerloop,output_path):
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
		temp = np.ones((r,cluster_no)) - np.matmul(image_2d_normalized,endmem_normalized) + mu * eu_distance(image_2d, endmem.transpose())**0.5
		f = 0.5*lamda*temp**2
		
		for jj in xrange(0,innerloop):
			count += 1
			uold = np.array(u, copy=True)
			data_inputs = [0 for i in xrange(0,cluster_no)]
			for l in xrange(0,cluster_no):
				data_inputs[l] = [u_bar[:,l],sigma,tao,W_sqrt,l,r,p[l],f[:,l],u[:,l]]
			# for l in xrange(0,cluster_no):
			# 	p[l] = p[l] + nonlocal_grad(W_sqrt, sigma*u_bar[:,l])
			# 	p[l] = project_p(p[l],r)
			# 	u[:,l] = u[:,l] + nonlocal_divergence(W_sqrt, tao*p[l]) - tao*f[:,l]
			pool = Pool(maxconn) # on 4 processors
			outputs = pool.map(calculate_cluster_wise_pdhg, data_inputs)
			pool.close()
			pool.join()
			for l in xrange(0,cluster_no):
				l_th_out = outputs[l]
				p[l] = l_th_out[0]
				u[:,l] = l_th_out[1]
			u = cond_projsplx_multi(u,(tao*f+1)**0.5)
			diff = LA.norm(u-uold,'fro')/LA.norm(u,'fro')
			u_bar = u+theta*(u-uold)
			error[count-1] = diff
		start_time = timeit.default_timer()
		uhard = threshold_u(u.transpose())
		print(timeit.default_timer() - start_time),"threshold_u execution time"
		endmem = calculate_centroid(uhard,cluster_no,endmem,image_2d)
		print outer_index
		L = assing_classes(uhard,m,n)
		ih.save_image(L,output_path + "_" + str(outer_index) + ".jpeg")
		stop = get_stop(uhard_old,uhard,r)
		uhard_old = uhard
	return error
