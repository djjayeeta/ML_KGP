import numpy as np
import pickle,random,os,time,random,sys
from cython.parallel import prange, parallel
from cython import boundscheck, wraparound
import image_helper as ih

@boundscheck(False)
@wraparound(False)
cdef double cabs(double x) nogil:
	if x >= 0.0:
		return x
	return x * (-1.0)

def get_initial_u(data, cluster_number):
	row_size = data.shape[0]
	col_size = data.shape[1]
	U = np.random.rand(row_size,col_size,cluster_number)
	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			total_u = sum(U[i][j])
			for k in xrange(0,cluster_number):
				U[i][j][k] = U[i][j][k] / total_u
	return U

def get_cluster_prototypes(U,data,m,cluster_number):
	row_size = data.shape[0]
	col_size = data.shape[1]
	channel_count = data.shape[2]
	V = np.zeros((cluster_number,channel_count))
	for r in xrange(0,cluster_number):
		normalizer = 0.0
		cluster_center = np.zeros(channel_count)
		for i in xrange(0,row_size):
			for j in xrange(0,col_size):
				cluster_center += (U[i][j][r]**m) * data[i][j]
				normalizer += U[i][j][r]**m
		V[r] = cluster_center / normalizer
	return V

def get_alpha(n,error_list):
	e = 2.71828183
	w = 9.0
	e_avg_t = 0.001
	if len(error_list) > 6:
		avg_error = sum(error_list[-6:]) / 6
		if avg_error < e_avg_t:
			n0 = n
		else:
			n0 = 40
	else:
		n0 = 40
	return 0.2 / (0.1 + e**((n0-n)/w))

def get_segmentation_error(L,L_new,data):
	row_size = data.shape[0]
	col_size = data.shape[1]

	total_error = 0.0
	count = 0
	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			if L_new[i][j] != L[i][j]:
				total_error += 1.0
			count += 1
	return total_error / float(count)

@boundscheck(False)
@wraparound(False)
cdef double get_feature_dissimilarity(double [:,:,:] X,int x,int y,double [:,:] V, int r, int channel_count) nogil:
	cdef double s = 0.0
	cdef int i
	for i in xrange(0,channel_count):
		s += (X[x][y][i] - V[r][i]) * (X[x][y][i] - V[r][i])
	return s**0.5

@boundscheck(False)
@wraparound(False)
cdef void compute_cluster_distances(double [:,:,:] U,double [:,:] V,double [:,:,:] X,double [:,:,:] D,int x,int y,int row_size,int col_size,int cluster_number,int channel_count,double alpha) nogil:
	cdef int i,j,r
	cdef double e = 2.71828183
	cdef double theta = 0.7
	cdef double beta = 0.0
	cdef double normalizer = 0.0
	
	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			beta = 1.0 / (   1.0 + e**( theta * (cabs(i - x) + cabs(j - y))  )   )
			for r in xrange(0,cluster_number):
				D[x][y][r] += U[i][j][r] * beta
				normalizer += U[i][j][r] * beta

	for r in xrange(0,cluster_number):
		D[x][y][r] = (1.0 - (D[x][y][r] / normalizer)) * alpha + get_feature_dissimilarity(X,x,y,V,r,channel_count)

	return 

@boundscheck(False)
@wraparound(False)
def get_dissimilarity_matrix(double [:,:,:]U,double [:,:]V,double [:,:,:]X,n,error_list):
	cdef int row_size = X.shape[0]
	cdef int col_size = X.shape[1]
	cdef int channel_count = X.shape[2]
	cdef double alpha = get_alpha(n,error_list)
	cdef int i = 0,j = 0,k = 0,l = 0
	cdef int cluster_number = V.shape[0]
	cdef double [:,:,:] D = np.zeros((row_size,col_size,cluster_number)) 
	cdef int [:,:] index_arr = np.array([[k,l] for k in xrange(row_size) for l in xrange(col_size)],dtype='int32')
	with nogil,parallel(num_threads = 8):
		for i in prange(row_size*col_size, schedule = "static"):
			compute_cluster_distances(U,V,X,D,index_arr[i][0],index_arr[i][1],row_size,col_size,cluster_number,channel_count,alpha)
	return np.array(D)

def update_U(U,D,m):
	row_size = U.shape[0]
	col_size = U.shape[1]
	cluster_number = U.shape[2]
	min_distance = 0.00000001

	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			good_classes = [c for c in xrange(0,cluster_number) if D[i][j][c] <= min_distance]
			if len(good_classes) > 0:
				for r in xrange(0,cluster_number):
					U[i][j][r] = 0.0
				for r in good_classes:
					U[i][j][r] = 1.0
			else:
				for r in xrange(0,cluster_number):
					U[i][j][r] = 1.0 / sum(    map(lambda x: ( D[i][j][r] / x )**(2 / (m-1)) ,   D[i][j])   )

	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			normalizer = sum(U[i][j])
			for r in xrange(0,cluster_number):
				U[i][j][r] = U[i][j][r] / normalizer

	return

def assing_classes(U):
	row_size = U.shape[0]
	col_size = U.shape[1]
	cluster_number = U.shape[2]

	L = [[0 for j in xrange(0,col_size)] for i in xrange(0,row_size)]

	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			L[i][j] = U[i][j].argmax()

	return L


def segment(pickle_data_file,cluster_number,output_path):
	data = ih.get_pickle_object_as_numpy(pickle_data_file)
	row_size = data.shape[0]
	col_size = data.shape[1]

	##### hyperparameters ########
	m = 1.3
	terminating_mean_error = 0.0002


	##### initializing ###########
	U = get_initial_u(data,cluster_number)
	error_list = []
	L_new = assing_classes(U)
	ih.save_image(L_new,output_path + "_" + str(0) + ".jpeg")

	##### starting iterations ####
	n = 1
	while(True):
		V = get_cluster_prototypes(U,data,m,cluster_number)
		ih.save_output(L_new,V,output_path + "_" + str(n) + ".pickle")
		D = get_dissimilarity_matrix(U,V,data,n,error_list)
		update_U(U,D,m)
		L = L_new
		L_new = assing_classes(U)
		ih.save_image(L_new,output_path + "_" + str(n) + ".jpeg")
		mean_error = get_segmentation_error(L,L_new,data)
		error_list.append(mean_error)

		if mean_error < terminating_mean_error:
			break
		n += 1
	return

