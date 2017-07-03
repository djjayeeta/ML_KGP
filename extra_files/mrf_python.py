from scipy import optimize
import numpy as np
import random,math,sys
import numpy.matlib
from helper import image_helper as ih
import timeit,math
from helper.jd_simanneal import Annealer
from scipy.cluster.vq import vq, kmeans, whiten
# np.seterr(divide='ignore', invalid='ignore')

class FunctionMinimizer(Annealer):
	def __init__(self, initial_state,data,mean,std_dev):
		self.mean = mean
		self.std_dev = std_dev
		cluster_number = mean.shape[0]
		self.cluster_number = cluster_number
		self.class_change_prob = 0.4
		self.alpha = 8
		self.beta = 4
		self.data = data
		self.neighboorhood = 16
		super(FunctionMinimizer, self).__init__(initial_state)

	def move(self):
		"""Random generator"""
		row_size = self.state.shape[0]
		col_size = self.state.shape[1]
		for i in xrange(0,row_size):
			for j in xrange(0,col_size):
				if self.class_change_prob > random.random():
					self.state[i][j] = random.randint(0,self.cluster_number-1)


	def calculate_Er_pixel_wise(self,i,j):
		radius = int(self.neighboorhood**0.5)
		row_size = self.data.shape[0]
		col_size = self.data.shape[1]
		pixel_er = 0.0
		for m in xrange(i - radius,i + radius):
			for n in xrange(j - radius,j + radius):
				if (m,n) != (i,j) and m>=0 and m < row_size and n >= 0 and n < col_size and (i-m)**2 + (j-n)**2 <= self.neighboorhood :
					delta = -1 if self.state[m][n] == self.state[i][j] else 1
					pixel_er += delta
		return self.beta * pixel_er

	def calculate_Ef_pixel_wise(self,i,j):
		cluster_no = self.state[i][j]
		vfunc = np.vectorize(math.log)
		std_dev = 2*self.std_dev[cluster_no]**2
		std_dev[std_dev==0] = 1
		pixel_ef_arr = ((self.data[i][j] - self.mean[cluster_no])**2) / std_dev
		log_std_dev = (2*math.pi)**0.5*self.std_dev[cluster_no]
		log_std_dev[log_std_dev<=0] = sys.float_info.min
		log_arr = vfunc(log_std_dev)
		pixel_ef_arr = pixel_ef_arr + log_arr
		return np.sum(pixel_ef_arr) 

	
	def energy(self):
		"""Calculates the length of the route."""
		row_size = self.data.shape[0]
		col_size = self.data.shape[1]
		E = 0.0
		for i in xrange(0,row_size):
			for j in xrange(0,col_size):
				E += self.calculate_Er_pixel_wise(i,j)  + self.alpha *  self.calculate_Ef_pixel_wise(i,j)
		return E

def minimize_energy(mean,std_dev,data,Y):
	cluster_number = mean.shape[0]
	fmin = FunctionMinimizer(Y,data,mean,std_dev)
	fmin.steps = 1000
	fmin.copy_strategy = "deepcopy"
	start_time = timeit.default_timer()
	e = fmin.energy()
	end_time = timeit.default_timer()
	print  end_time - start_time,e
	Y_new, e = fmin.anneal()
	# print()
	# print("%f function value x:" % e)
	# print("\t", Y_new)
	return Y_new

def get_initial_seg(row_size,col_size, cluster_number,data):
	r,channel = row_size*col_size,data.shape[2]
	X = np.reshape(data,(r,channel))
	kmeans_res = kmeans(X,cluster_number)
	V = kmeans_res[0].astype(dtype=np.float64)
	Y = np.random.randint(cluster_number,size=(row_size,col_size))
	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			z = (((np.matlib.repmat(data[i][j],cluster_number,1) - V)**2).sum(axis=1)) ** 0.5
			z = z/np.sum(z)
			z_exp = [math.exp(-k) for k in z]  
			sum_z_exp = sum(z_exp)  
			softmax = [round(k / sum_z_exp, 3) for k in z_exp]
			Y[i][j] = np.argmax(np.array(softmax))
	return Y

def get_cluster_prototypes(Y,data,cluster_number):
	row_size = data.shape[0]
	col_size = data.shape[1]
	channel_count = data.shape[2]
	mean = np.zeros((cluster_number,channel_count),dtype=np.float64)
	count_sample = [0 for i in xrange(cluster_number)]
	std_dev = np.zeros((cluster_number,channel_count),dtype=np.float64)

	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			mean[Y[i][j]] = mean[Y[i][j]] + data[i][j]
			count_sample[Y[i][j]] += 1

	for i in xrange(0,cluster_number):
		if count_sample[i] != 0:
			mean[i] = mean[i]/count_sample[i]

	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			std_dev[Y[i][j]] = std_dev[Y[i][j]] + ((data[i][j] - mean[Y[i][j]])**2)

	for i in xrange(0,cluster_number):
		if count_sample[i]-1!=0:
			std_dev[i] = (std_dev[i]/(count_sample[i]-1))**0.5
	# print std_dev[~np.isfinite(std_dev)]
	# print mean[~np.isfinite(mean)]

	return mean,std_dev

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

def segment(pickle_data_file,cluster_number,output_path):
	data = ih.get_pickle_object_as_numpy(pickle_data_file)
	row_size = data.shape[0]
	col_size = data.shape[1]

	##### hyperparameters ########
	m = 1.3
	terminating_mean_error = 0.0002


	##### initializing ###########
	Y_new = get_initial_seg(row_size,col_size,cluster_number,data)
	ih.save_image(Y_new,output_path + "_" + str(0) + ".jpeg")

	##### starting iterations ####
	n = 1
	while(True):
		#### E step ####
		mean,std_dev  = get_cluster_prototypes(Y_new,data,cluster_number)
		ih.save_output(Y_new,{'mean':mean,'std_dev':std_dev},output_path + "_" + str(n) + ".pickle")

		#### M Step###

		Y = np.array(Y_new, copy=True)
		Y_new = minimize_energy(mean,std_dev,data,Y_new)


		ih.save_image(Y_new,output_path + "_" + str(n) + ".jpeg")
		mean_error = get_segmentation_error(Y,Y_new,data)
		
		if mean_error < terminating_mean_error:
			break

		n += 1
	return

