from osgeo import gdal
import numpy as np
import pickle,random,os
from PIL import Image
from math import exp,pow,sqrt
from multiprocessing import Pool
from datetime import datetime
import timeit
import threading
import time, random,sys

MAX = 1000.0
Epsilon = 0.0002
EpsilonAvg = 0.001
Theta = 0.7
n0 = 40
last_six = [0,0,0,0,0,0]
w = 0.9
fuzzy_index = 1.3
fuzzy_index_prev = 1.3
neg_dist = 0
cluster_number = 6
cluster_color = {0:[255,255,0],1:[128,255,0],2:[0,128,255],3:[255,0,255],4:[255,0,0],5:[0,0,0]}
image_path = "IMS1_HYSI_GEO_114_05FEB2009_S1_TOA_REFLECTANCE_07_SPBIN.tif"
maxconn=8
pool=threading.BoundedSemaphore(value=maxconn)
dist_mat = []


def calculate_fuzzy_index(itertion_number,h,fxkyk):
	fuzzy_index = fuzzy_index_prev + h*fxkyk
	fuzzy_index_prev = fuzzy_index
	return fuzzy_index


def get_data_from_image():
	global image_path
	if image_path.split(".")[1] == "tif":
		dataset = gdal.Open(image_path,gdal.GA_ReadOnly)
		col = dataset.RasterXSize
		row = dataset.RasterYSize
		a = [[[]for y in xrange(col)] for z in xrange(row)]
		for i in xrange(1,dataset.RasterCount + 1):
			band = dataset.GetRasterBand(i).ReadAsArray()
			for m in xrange(0,row):
				for n in xrange(0,col):
					a[m][n].append(band[m][n])
		M = np.array(a,dtype='uint16')
	else:
		M = np.asarray(Image.open(image_path))
	image_name = image_path.split(".")[0]
	f = open("data_"+str(image_name)+".pickle","wb")
	pickle.dump(M, f)
	f.close()

def create_image():
	row = 50
	col = 50
	channel = 3
	global image_path
	a = [[[0 for x in xrange(0,channel)]for y in xrange(0,col)] for z in xrange(0,row)]
	for x in xrange(0,row):
		for y in xrange(0,col):
			if x <= row/2 and y <= col/2:
				a[x][y] = [255,0,0]
			elif x <= row/2 and y > col/2:
				a[x][y] = [0,0,255]
			elif x > row/2 and y <= col/2:
				a[x][y] = [0,255,0]
			else:
				a[x][y] = [0,0,0]
	a = np.array(a,dtype='uint8')
	im = Image.fromarray(a)
	im.save(image_path)


def save_image_from_L(L,itertion_number,datetime_str):
	global cluster_number,image_path
	image_name = image_path.split(".")[0]
	if not os.path.exists(image_name+"_"+datetime_str):
		os.makedirs(image_name+"_"+datetime_str)
	a = [[[] for y in xrange(0,len(L[0]))] for z in xrange(0,len(L))]
	for x in xrange(0,len(L)):
		for y in xrange(0,len(L[0])):
			a[x][y] = cluster_color[L[x][y]]
	a = np.array(a,dtype='uint8')
	im = Image.fromarray(a)
	im.save(image_name+"/"+image_name+"_"+str(itertion_number)+".jpeg")

def get_data_from_pickle():
	global image_path
	image_name = image_path.split(".")[0]
	f = open("data_"+image_name+".pickle","r")
	irrad = pickle.load(f)
	f.close()
	return irrad

def initialise_U(data, cluster_number):
	global MAX
	row = len(data)
	col = len(data[0])
	U = [[[0 for z in xrange(0,cluster_number) ] for y in xrange(0,col)]for z in xrange(0,row)]
	
	for i in range(0,row):
		for j in range(0,col):
			rand_sum = 0.0
			for k in range(0,cluster_number):
				dummy = random.randint(1,int(MAX))
				U[i][j][k] = dummy
				rand_sum += dummy
			for k in range(0,cluster_number):
				U[i][j][k] = U[i][j][k] / rand_sum
	U = np.array(U)
	return U

def delta(val1,val2):
	if abs(val1 - val2) <= 0.000000001:
		return 0
	return 1

def end_conditon(L,L_old):
	global Epsilon,last_six
	total_error = 0.0
	row,col = len(L),len(L[0])
	for i in xrange(0,row):
		for j in xrange(0,col):
			total_error += delta(L[i][j], L_old[i][j])
	error_frac = float(total_error/float(row*col))
	last_six = [last_six[1],last_six[2],last_six[3],last_six[4],last_six[5],error_frac]
	print error_frac,'error_frac'
	if error_frac < Epsilon:
		return True
	return False

def normalise_U(U,L):
	row = len(U)
	col = len(U[0])
	cluster_number = len(U[0][0])
	for i in xrange(0,row):
		for j in xrange(0,col):
			maximum = max(U[i][j])
			for k in range(0,cluster_number):
				if U[i][j][k] == maximum:
					L[i][j] = k
					break
	return L

def get_alpha(itertion_number):
	global n0,last_six,w
	avg = sum(last_six)/6
	if avg < EpsilonAvg:
		n0 = itertion_number	
	x = (itertion_number - n0)/w
	# print x,'x'
	alpha = 0.2 / (0.1 + exp(-x))
	return alpha


def neigh_contri_by_point(sitex,sitey,pointx,pointy):
	global Theta
	dist = Theta*eu_distance([sitex,sitey],[pointx,pointy])
	return float(1)/float(1+exp(dist))

def cluster_spatial_dist(params):
	sitex,sitey,cluster_index,U = params[0],params[1],params[2],params[3]
	total = 0.0
	row = len(U)
	col = len(U[0])
	# print cluster_index
	for m in xrange(0,row):
		for n in xrange(0,col):
			betat = neigh_contri_by_point(sitex,sitey,n,m)
			# print betat,'betat'
			if not betat < 0.001:
				# print betat,'betataccepted'
				total += U[m][n][cluster_index] * betat
	return total


def spatial_dist_denom(sitex,sitey,U,itertion_number):
	global maxconn
	alpha = get_alpha(itertion_number)
	# print alpha,'alpha'
	if alpha >0.00002:
		cluster_number = len(U[0][0])
		cluster_spatial_arr = [0.0 for j in xrange(0,cluster_number)]
		data_inputs = [[sitex,sitey,j,U] for j in xrange(0,cluster_number)]
		for j in xrange(0,cluster_number):

		# pool=threading.BoundedSemaphore(value=maxconn)
		# pool = Pool(4) # on 4 processors
		# cluster_spatial_arr = pool.map(cluster_spatial_dist, data_inputs)
		# pool.close()
		# pool.join()

			cluster_spatial_arr[j] = cluster_spatial_dist(data_inputs[j])
		return cluster_spatial_arr,alpha
	return [],0

def spatial_distance(cluster_index,cluster_spatial_arr,alpha):
	if sum(cluster_spatial_arr) == 0.0:
		return 0
	return alpha*(1.0 - float(cluster_spatial_arr[cluster_index]/sum(cluster_spatial_arr)))

def eu_distance(point,center):
	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0,len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	return sqrt(dummy)

def initialize_v(v,v_denom):
	for i in xrange(0,len(v_denom)):
		v_denom[i] = 0.0
		for j in xrange(0,len(v[0])):
			v[i][j] = 0.0
	return v,v_denom

def get_cluster_ref(U,data,v,v_denom):
	global fuzzy_index
	cluster_number = len(U[0][0])
	row = len(data)
	col = len(data[0])
	channel = len(data[0][0])
	v,v_denom = initialize_v(v,v_denom)
	total_denominator = 0.0
	for k in xrange(0,cluster_number):
		for m in xrange(0,row):
			for n in xrange(0,col):
				denom = pow(U[m][n][k],fuzzy_index)
				# print denom,U[m][n][k]
				v_denom[k] += denom
				for l in xrange(0,channel):
					v[k][l] += denom*data[m][n][l]
	for k in xrange(0,cluster_number):
		for l in xrange(0,channel):
			v[k][l] /= v_denom[k]
	return v

def calculate_distance(params):
	global cluster_number
	# global dist_mat
	# print dist_mat
	n,m,U,itertion_number,data,cluster_centres = params[0],params[1],params[2],params[3],params[4],params[5]
	# print n,m
	# cluster_number = len(cluster_centres)
	cluster_spatial_arr,alpha = spatial_dist_denom(n,m,U,itertion_number)
	dist = [0 for i in xrange(0,cluster_number)]
	for r in xrange(0,cluster_number):
		dist[r] = eu_distance(data[m][n],cluster_centres[r]) + spatial_distance(r,cluster_spatial_arr,alpha)
		# if dist != dist_mat[m][n][r]:
		# 	print 'yes',m,n,r
		# dist_mat[m][n][r] = dist
	return dist

class GrabUrl(threading.Thread):
	def __init__(self, arg0):
		threading.Thread.__init__(self)
		self.params = arg0
	def run(self):
		calculate_distance(self.params)
		pool.release()

class Handler(threading.Thread):
	def run(self):
		global pool
		data_inputs = self._Thread__kwargs['data_inputs']
		for i in xrange(0,len(data_inputs)):
			pool.acquire()
			graburl = GrabUrl(data_inputs[i])
			graburl.setDaemon(True)
			graburl.start()

def update_U(U,data,itertion_number,cluster_centres,dist_mat):
	global neg_dist,fuzzy_index,maxconn
	row = len(data)
	col = len(data[0])
	cluster_number = len(cluster_centres)
	k = 0
	data_inputs = [[] for i in xrange(0,row*col)]
	for m in xrange(0,row):
		for n in xrange(0,col):
			# print dist_mat[m][n]
			# time.sleep(1)
			data_inputs[(m*col)+n] = [n,m,U,itertion_number,data,cluster_centres]
	# handler = Handler(kwargs={'data_inputs':data_inputs})
	# handler.start()
	# handler.join()
	# cluster_spatial_arr

	pool = Pool(maxconn) # on 4 processors
	outputs = pool.map(calculate_distance, data_inputs)
	pool.close()
	pool.join()
	
	for m in xrange(0,row):
		for n in xrange(0,col):
			dist_mat[m][n] = outputs[m*col+n] 
		


	# 		cluster_spatial_arr,alpha = spatial_dist_denom(n,m,U,itertion_number)
	# 		for r in xrange(0,cluster_number):
	# 			k += 1
	# 			dist = eu_distance(data[m][n],cluster_centres[r]) + spatial_distance(r,cluster_spatial_arr,alpha)
	# 			if k%1000 == 0:
	# 			# if k == 2000:
	# 				# return
	# 				print 'dist',k
	# 			dist_mat[m][n][r] = dist
	for m in xrange(0,row):
		for n in xrange(0,col):
			# print dist_mat[m][n]
			indices = [i for i in xrange(0,cluster_number) if dist_mat[m][n][i] <= neg_dist ]
			for r in xrange(0,cluster_number):
				if indices and r in indices:
					U[m][n][r] = 1
				elif indices and r not in indices:
					U[m][n][r] = 0
				else:
					U[m][n][r] = 1/sum([pow(dist_mat[m][n][r]/dist_mat[m][n][k],float(2/fuzzy_index-1)) for k in xrange(0,cluster_number)])
	return U

def get_L(U):
	row = len(U)
	col = len(U[0])
	L = [[0 for x in xrange(0,col)] for y in xrange(0,row)]
	L = np.array(L)
	return L


def fuzzy_cluster(cluster_number,datetime_str):
	global dist_mat
	data = get_data_from_pickle()
	# print data,'data'
	row = len(data)
	col = len(data[0])
	itertion_number = 1
	U = initialise_U(data,cluster_number)
	dist_mat = [[[0.0 for x in xrange(0,cluster_number)] for y in xrange(0,col)] for z in xrange(0,row)]
	dist_mat = np.array(dist_mat)
	# print U,'U'
	cluster_centres = [[0.0 for y in xrange(len(data[0][0]))] for x in xrange(0,cluster_number)]
	v_denom = [0.0 for x in xrange(0,cluster_number)]
	L = get_L(U)
	L_new = get_L(U)
	L = normalise_U(U,L)
	save_image_from_L(L,itertion_number,datetime_str)
	# print L,'L'
	while True:
		cluster_centres = get_cluster_ref(U,data,cluster_centres,v_denom)
		print cluster_centres,'cluster_centres'
		U = update_U(U,data,itertion_number,cluster_centres,dist_mat)
		# return
		# print U,'U'
		print itertion_number,'itertion_number'
		L_new = normalise_U(U,L_new)
		to_end = end_conditon(L_new,L)
		L = np.array(L_new, copy=True)
		itertion_number += 1
		save_image_from_L(L,itertion_number,datetime_str)
		if to_end:
			break
	L = normalise_U(U,L)
	return L

def run_fuzzy():
	datetime_str = str(datetime.now())
	global cluster_number
	# create_image()
	start_time = timeit.default_timer()
	get_data_from_image()
	get_data_from_pickle()
	L = fuzzy_cluster(cluster_number,datetime_str)
	print(timeit.default_timer() - start_time),"total execution time"
	# print L

	f = open("seg_"+image_name+".pickle","wb")
	pickle.dump(L, f)
	f.close()

image_path = sys.argv[1]
run_fuzzy()

