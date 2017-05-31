from scipy import optimize
import numpy as np
import pickle,random,math
# from cython.parallel import prange, parallel
# from cython import boundscheck, wraparound
import image_helper as ih
import timeit
from jd_simanneal import Annealer


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
		pixel_ef_arr = ((self.data[i][j] - self.mean[cluster_no])**2) / (2*self.std_dev[cluster_no]**2)
		log_arr = vfunc((2*math.pi)**0.5*self.std_dev[cluster_no])
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
	fmin.steps = 10000
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

def get_initial_seg(row_size,col_size, cluster_number):
	Y = np.random.randint(cluster_number,size=(row_size,col_size))
	return Y

def get_cluster_prototypes(Y,data,cluster_number):
	row_size = data.shape[0]
	col_size = data.shape[1]
	channel_count = data.shape[2]
	mean = np.zeros((cluster_number,channel_count))
	count_sample = [0 for i in xrange(cluster_number)]
	std_dev = np.zeros((cluster_number,channel_count))

	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			mean[Y[i][j]] = mean[Y[i][j]] + data[i][j]
			count_sample[Y[i][j]] += 1

	for i in xrange(0,cluster_number):
		mean[i] = mean[i]/count_sample[i]

	for i in xrange(0,row_size):
		for j in xrange(0,col_size):
			std_dev[Y[i][j]] = std_dev[Y[i][j]] + ((data[i][j] - mean[Y[i][j]])**2)

	for i in xrange(0,cluster_number):
		std_dev[i] = (std_dev[i]/(count_sample[i]-1))**0.5

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
	Y_new = get_initial_seg(row_size,col_size,cluster_number)
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

