from pdhg_linear import pd_nonlocal_HSI
from make_weight import weights_HSI
import numpy as np
from sklearn.cluster import KMeans
import scipy.io
import timeit


def get_cluster_centroids(data,cluster_number):
	r,channel = data.shape[0]*data.shape[1],data.shape[2]
	X = np.reshape(data,(r,channel))
	kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(X)
	return kmeans.cluster_centers_.transpose()
	
def create_weight_centroid_mat_file(image,output_path,cluster_number):
	W = weights_HSI(image)
	endmem_rand = get_cluster_centroids(image,cluster_number)
	scipy.io.savemat(output_path+".mat",{'W_binary_10':W,'H':image,'endmem_rand':endmem_rand})
	return W,endmem_rand

def run_linear_pdhg(mat_file_path,create_weight,output_path,cluster_number,start_time):
	mat = scipy.io.loadmat(mat_file_path)
	image = mat['H']
	if create_weight:
		W,endmem_rand = create_weight_centroid_mat_file(image,output_path+"_data",cluster_number)
		print (timeit.default_timer() - start_time),"weight execution time"
	else:
		W = mat['W_binary_10']
		endmem_rand = mat['endmem_rand']
	mu = 1e-5
	lamda = 1e6
	tao = 10.0
	sigma = 1.0/tao/4.0/10.0
	sigma = sigma/3
	theta = 1
	tol = 1e-2
	iter_stop = 1.1
	innerloop = 5
	outerloop = 50
	error = pd_nonlocal_HSI(image,W,mu,endmem_rand,lamda,tao,sigma,theta,tol,iter_stop,innerloop,outerloop,output_path)
	return error

def local_run():
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
	output_path = "data_old/linear_hybrid_output_"
	endmem = endmem_rand
	image = f
	error = pd_nonlocal_HSI(f,W,mu,endmem_rand,lamda,tao,sigma,theta,tol,iter_stop,innerloop,outerloop,output_path)
	
start_time = timeit.default_timer()
cluster_number = 6
error = run_linear_pdhg("data_urban.mat",True,"data/linear_hybrid_output_",cluster_number,start_time)
print (timeit.default_timer() - start_time),"total execution time"
print error
