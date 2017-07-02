import numpy as np
import numpy.matlib
from scipy import sparse
import pickle,random,os,time,random,sys,math
from cython.parallel import prange, parallel
from cython import boundscheck, wraparound
from helper import image_helper as ih
import itertools,pickle,timeit
from scipy.cluster.vq import vq, kmeans, whiten
from multiprocessing import Pool
maxconn = 8


def get_cluster_centroids(data,cluster_number):
	data_new = data.astype(dtype=np.float64)
	r,channel = data.shape[0]*data.shape[1],data.shape[2]
	X = np.reshape(data_new,(r,channel))
	kmeans_res = kmeans(X,cluster_number)
	return kmeans_res[0].astype(dtype=np.float64)

def get_beta(row_size,col_size,beta_pickle_path,compute_beta):
	if compute_beta:
		theta = 0.7
		e = 2.71828183
		max_no_of_points = 500
		r = row_size*col_size
		all_points = np.arange(0,r)
		all_points_row = all_points/col_size
		all_points_col = all_points%col_size
		cols = [[] for i in xrange(0,r)]
		rows = [[] for i in xrange(0,r)]
		values = [[] for i in xrange(0,r)]
		# start_time = timeit.default_timer()
		for i in xrange(0,r):
			i_row,i_col = i/col_size,i%col_size
			values_i = (abs(all_points_row - i_row) +  abs(all_points_col-i_col))
			values_i = 1.0/(1.0 + e**(theta*values_i))
			values_i[values_i<1e-7] = 0
			col_indexes = np.nonzero(values_i>0)[0]    
			if max_no_of_points !=-1 and col_indexes.shape > max_no_of_points:
				col_indexes = np.argsort(values_i)[::-1][:max_no_of_points]
			cols[i] = (col_indexes).tolist()
			row_indexes = np.zeros(col_indexes.shape[0],dtype=np.int32)+i
			rows[i] = row_indexes.tolist()
			values[i] = (values_i[col_indexes]).tolist()
			del values_i
			del col_indexes
			del row_indexes
		cols = list(itertools.chain(*cols))
		rows = list(itertools.chain(*rows))
		values = list(itertools.chain(*values))
		beta = sparse.csr_matrix((values,(rows,cols)),shape = (r,r))
		with open(beta_pickle_path,"wb") as fp:
			pickle.dump(beta,fp, protocol=pickle.HIGHEST_PROTOCOL)
		return beta
	else:
		beta_pickle_path = "data/hsi_pickle_beta_new.pickle"
		with open(beta_pickle_path,"rb") as fp:
			beta = pickle.load(fp)
		#beta = beta.transpose()
	return beta

def get_initial_u(data, cluster_number):
	row_size = data.shape[0]
	col_size = data.shape[1]
	U = np.random.rand(row_size,col_size,cluster_number)
	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			total_u = sum(U[i][j])
			U[i][j] = U[i][j]/total_u
			#z = (((np.matlib.repmat(data[i][j],cluster_number,1) - V)**2).sum(axis=1)) ** 0.5
			#z = z/np.sum(z)
			#z_exp = [math.exp(-k) for k in z]  
			#sum_z_exp = sum(z_exp)  
			#softmax = [round(k / sum_z_exp, 3) for k in z_exp]
			#U[i][j] = np.array(softmax)
			# for k in xrange(0,cluster_number):
			# 	U[i][j][k] = U[i][j][k] / total_u
	return U


def get_cluster_prototypes(U,data,m,cluster_number,iteration_no):
	# if iteration_no == 1:
	# 	V = get_cluster_centroids(data,cluster_number)
	# else:
	row_size = data.shape[0]
	col_size = data.shape[1]
	channel_count = data.shape[2]
	V = np.zeros((cluster_number,channel_count))
	U_new = U.reshape(row_size*col_size,cluster_number)
	data_new = data.reshape(row_size*col_size,channel_count)
	normalizer = np.sum(U**m,axis=(0,1))
	for r in xrange(0,cluster_number):
		# normalizer = 0.0
		# cluster_center = np.zeros(channel_count)
		# for i in xrange(0,row_size):
		# 	for j in xrange(0,col_size):
		# 		cluster_center += (U[i][j][r]**m) * data[i][j]
		# 		normalizer += U[i][j][r]**m
		V[r] = ((np.matlib.repmat(U_new[:,r]**m,channel_count,1).transpose() * data_new).sum(axis=0))/normalizer[r]
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


def get_feature_dissimilarity(X,x,y,V,r,channel_count):
	# s = 0.0
	s = np.sum((X[x][y] - V[r])**2,axis=0)
	# cdef int i
	# for i in xrange(0,channel_count):
	# 	s += (X[x][y][i] - V[r][i]) * (X[x][y][i] - V[r][i])
	return s**0.5

def compute_cluster_distances_pool(params):
	U_new,V,X_x_y,x,y,alpha,beta_x_y = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
	# row_size = X.shape[0]
	# col_size = X.shape[1]
	# channel_count = X.shape[2]
	cluster_number = U_new.shape[1]
	# i,j,r
	# beta = 0.0
	normalizer = 0.0
	# U_new = U.reshape(row_size*col_size,cluster_number)
	D = np.sum(U_new.transpose() * np.matlib.repmat(beta_x_y.toarray()[0],cluster_number,1),axis=1)
	sum_D = np.sum(D)
	if sum_D == 0:
		sum_D = 1
	D = (1.0 - (D/sum_D)) * alpha + np.sum((np.matlib.repmat(X_x_y,cluster_number,1) - V)**2,axis=1)**0.5

	# for j in xrange(0,cluster_number):
	# 	D[x][y][j] = np.sum(U_new[:,j]*beta[x*row_size+y,:].toarray()[0])
	# 	normalizer += D[x][y][j]

	# # for i in xrange(0,row_size):
	# 	# for j in xrange(0,col_size):
	# 	# 	beta = 1.0 / (   1.0 + e**( theta * (cabs(i - x) + cabs(j - y))  )   )
	# 	# 	for r in xrange(0,cluster_number):
	# 	# 		D[x][y][r] += U[i][j][r] * beta
	# 	# 		normalizer += U[i][j][r] * beta

	# for r in xrange(0,cluster_number):
	# 	D[x][y][r] = (1.0 - (D[x][y][r] / normalizer)) * alpha + get_feature_dissimilarity(X,x,y,V,r,channel_count)

	return D

def compute_cluster_distances(U,V,X,D,x,y,alpha,beta):
	row_size = X.shape[0]
	col_size = X.shape[1]
	channel_count = X.shape[2]
	cluster_number = U.shape[2]
	# i,j,r
	# beta = 0.0
	normalizer = 0.0
	U_new = U.reshape(row_size*col_size,cluster_number)
	D[x][y] = np.sum(U_new.transpose() * np.matlib.repmat(beta[x*row_size+y,:].toarray()[0],cluster_number,1),axis=1)

	D[x][y] = (1.0 - (D[x][y]/np.sum(D[x][y]))) * alpha + np.sum((np.matlib.repmat(X[x][y],cluster_number,1) - V)**2,axis=1)**0.5

	# for j in xrange(0,cluster_number):
	# 	D[x][y][j] = np.sum(U_new[:,j]*beta[x*row_size+y,:].toarray()[0])
	# 	normalizer += D[x][y][j]

	# # for i in xrange(0,row_size):
	# 	# for j in xrange(0,col_size):
	# 	# 	beta = 1.0 / (   1.0 + e**( theta * (cabs(i - x) + cabs(j - y))  )   )
	# 	# 	for r in xrange(0,cluster_number):
	# 	# 		D[x][y][r] += U[i][j][r] * beta
	# 	# 		normalizer += U[i][j][r] * beta

	# for r in xrange(0,cluster_number):
	# 	D[x][y][r] = (1.0 - (D[x][y][r] / normalizer)) * alpha + get_feature_dissimilarity(X,x,y,V,r,channel_count)

	return 


def get_dissimilarity_matrix(U,V,X,n,error_list,beta):
	row_size = X.shape[0]
	col_size = X.shape[1]
	channel_count = X.shape[2]
	alpha = get_alpha(n,error_list)
	# i = 0,j = 0,k = 0,l = 0
	cluster_number = V.shape[0]
	D = np.zeros((row_size,col_size,cluster_number)) 

	index_arr = np.array([[k,l] for k in xrange(row_size) for l in xrange(col_size)],dtype='int32')
	# with nogil,parallel(num_threads = 8):
	U_new = U.reshape(row_size*col_size,cluster_number)
	data_inputs = [0 for i in xrange(0,row_size*col_size)]

	for i in xrange(0, row_size*col_size):
		x = index_arr[i][0]
		y = index_arr[i][1]
		data_inputs[i] = [U_new,V,X[x][y],x,y,alpha,beta[x*row_size+y,:]]
		# compute_cluster_distances(U,V,X,D,index_arr[i][0],index_arr[i][1],alpha,beta)
	pool = Pool(maxconn) 
	outputs = pool.map(compute_cluster_distances_pool, data_inputs)
	pool.close()
	pool.join()
	for i in xrange(0,row_size*col_size):
		x = index_arr[i][0]
		y = index_arr[i][1]
		D[x][y] = outputs[i]

	return D

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

	#V = get_cluster_centroids(data,cluster_number)
	##### initializing ###########
	U = get_initial_u(data,cluster_number)
	error_list = []
	L_new = assing_classes(U)
	ih.save_image(L_new,output_path + "_" + str(0) + ".jpeg")
	start_time = timeit.default_timer()
	beta = get_beta(row_size,col_size,pickle_data_file.split(".pickle")[0]+"_beta_new.pickle",False)
	print (timeit.default_timer() - start_time),"beta executiion time"
	##### starting iterations ####
	n = 1
	while(True):
		V = get_cluster_prototypes(U,data,m,cluster_number,1)
		ih.save_output(L_new,V,output_path + "_" + str(n-1) + ".pickle")
		start_time = timeit.default_timer()
		D = get_dissimilarity_matrix(U,V,data,n,error_list,beta)
		print (timeit.default_timer() - start_time),"D executiion time"
		update_U(U,D,m)
		L = L_new
		L_new = assing_classes(U)
		ih.save_image(L_new,output_path + "_" + str(n) + ".jpeg")
		mean_error = get_segmentation_error(L,L_new,data)
		error_list.append(mean_error)

		if mean_error < terminating_mean_error:
			break
		n += 1
	ih.save_output(L_new,V,output_path + "_" + str(n) + ".pickle")
	return
