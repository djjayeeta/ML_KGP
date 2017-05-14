from osgeo import gdal
import numpy as np
import pickle,random,os
from PIL import Image
from math import exp,pow,sqrt
from multiprocessing import Pool
from datetime import datetime
import timeit
import threading
from cython.parallel import prange, parallel
from cython import boundscheck, wraparound
import time, random,sys,cython



def calculate_fuzzy_index(itertion_number,h,fxkyk):
	global fuzzy_index,fuzzy_index_prev
	fuzzy_index = fuzzy_index_prev + h*fxkyk
	fuzzy_index_prev = fuzzy_index
	return fuzzy_index

def save_pickle_from_image(image_path):
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
		M = M.astype(np.float64)
	else:
		M = np.asarray(Image.open(image_path),dtype=np.float64)
		print M.shape
	image_name = image_path.split(".")[0]
	f = open("data_"+str(image_name)+".pickle","wb")
	pickle.dump(M, f)
	f.close()

def create_image(image_path):
	row = 100
	col = 100
	channel = 3
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


def save_image_from_L(L,itertion_number,datetime_str,cluster_color,image_path):
	image_name = image_path.split(".")[0]
	if not os.path.exists(image_name+"_"+datetime_str):
		os.makedirs(image_name+"_"+datetime_str)
	a = [[[] for y in xrange(0,len(L[0]))] for z in xrange(0,len(L))]
	for x in xrange(0,len(L)):
		for y in xrange(0,len(L[0])):
			a[x][y] = cluster_color[L[x][y]]
	a = np.array(a,dtype='uint8')
	im = Image.fromarray(a)
	im.save(image_name+"_"+datetime_str+"/"+image_name+"_"+str(itertion_number)+".jpeg")

def get_data_from_pickle(image_path):
	image_name = image_path.split(".")[0]
	f = open("data_"+image_name+".pickle","r")
	irrad = pickle.load(f)
	f.close()
	return irrad

def initialise_U(data, cluster_number,MAX):
	row = len(data)
	col = len(data[0])
	U = np.zeros((row,col,cluster_number))
	for i in range(0,row):
		for j in range(0,col):
			rand_sum = 0.0
			for k in range(0,cluster_number):
				dummy = random.randint(1,int(MAX))
				U[i][j][k] = dummy
				rand_sum += dummy
			for k in range(0,cluster_number):
				U[i][j][k] = U[i][j][k] / rand_sum
	return U

def delta(val1,val2):
	if abs(val1 - val2) == 0:
		return 0
	return 1

def end_conditon(L,L_old,row,col,Epsilon,last_six):
	total_error = 0.0
	row,col = len(L),len(L[0])
	for i in xrange(0,row):
		for j in xrange(0,col):
			total_error += delta(L[i][j], L_old[i][j])
	error_frac = float(total_error/float(row*col))
	last_six = np.array([last_six[1],last_six[2],last_six[3],last_six[4],last_six[5],error_frac])
	print error_frac,'error_frac'
	if error_frac < Epsilon:
		return True,last_six
	return False,last_six

def segment_image(U,L,row,col,cluster_number):
	for i in xrange(0,row):
		for j in xrange(0,col):
			maximum = max(U[i][j])
			for k in range(0,cluster_number):
				if U[i][j][k] == maximum:
					L[i][j] = k
					break
	return L

def get_n0(n0,last_six,EpsilonAvg,itertion_number):
	avg = sum(last_six)/6
	if avg < EpsilonAvg:
		n0 = itertion_number
	return n0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_alpha(int itertion_number,int n0,double [:] last_six,double w,double EpsilonAvg) nogil:
	cdef double x,alpha
	x = (itertion_number - n0)/w
	alpha = 0.2 / (0.1 + (2.71828183**(-x)))
	return alpha

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double neigh_contri_by_point(int sitex,int sitey,int pointx,int pointy,double Theta) nogil:
	cdef double dist,dist_ret,x_dist,y_dist
	x_dist = sitex - pointx 
	y_dist = sitey - pointy
	if x_dist < 0:
		x_dist = pointx - sitex 
	if y_dist < 0:
		y_dist = pointy - sitey
	dist = Theta*(x_dist + y_dist)
	dist_ret = 1/float(1+(2.71828183**dist))
	return dist_ret

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cluster_spatial_dist(int sitex,int sitey,int cluster_index,double [:,:,:] U,double Theta) nogil:
	cdef double betat,total = 0
	cdef int row,col,m,n
	row = U.shape[0]
	col = U.shape[1]
	for m in xrange(0,row):
		for n in xrange(0,col):
			betat = neigh_contri_by_point(sitex,sitey,n,m,Theta)
			if not betat < 0.001:
				total += U[m][n][cluster_index] * betat
	return total

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void spatial_dist_denom(int sitex,int sitey,double [:,:,:] U,int itertion_number,int n0,double [:] last_six,double w,double EpsilonAvg,double alpha,double [:] cluster_spatial_arr,int cluster_number,double Theta) nogil:
	cdef int j
	if alpha >0.00002:
		for j in xrange(0,cluster_number):
			cluster_spatial_arr[j] = cluster_spatial_dist(sitex,sitey,j,U,Theta)
	
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double spatial_distance(int cluster_index,double [:] cluster_spatial_arr,double alpha) nogil:
	cdef int x
	cdef double s = 0
	if alpha <= 0.00002:
		return 0.0
	for x in xrange(0,cluster_spatial_arr.shape[0]):
		s += cluster_spatial_arr[x]
	if s == 0.0:
		return 0.0
	return alpha*(1.0 - float(cluster_spatial_arr[cluster_index]/s))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double eu_distance(double [:] point,double [:] center , int point_len) nogil:
	cdef double dummy = 0
	cdef int i
	for i in range(0,point_len):
		dummy += (point[i] - center[i]) ** 2
	return (dummy)**0.5

def initialize_v(v,v_denom):
	for i in xrange(0,len(v_denom)):
		v_denom[i] = 0.0
		for j in xrange(0,len(v[0])):
			v[i][j] = 0.0
	return v,v_denom

def get_cluster_ref(U,data,cluster_centres,v_denom,fuzzy_index,row,col,cluster_number,channel):
	cluster_centres,v_denom = initialize_v(cluster_centres,v_denom)
	for k in xrange(0,cluster_number):
		for m in xrange(0,row):
			for n in xrange(0,col):
				denom = pow(U[m][n][k],fuzzy_index)
				v_denom[k] += denom
				for l in xrange(0,channel):
					cluster_centres[k][l] += denom*data[m][n][l]
	for k in xrange(0,cluster_number):
		for l in xrange(0,channel):
			cluster_centres[k][l] /= v_denom[k]
	return cluster_centres

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calculate_distance(int n,int m,int itertion_number,int cluster_number,double [:,:,:] U,double[:,:,:] data,double[:,:] cluster_centres,double[:,:,:] dist_mat,int n0,double[:] last_six,double w,double EpsilonAvg,double [:] cluster_spatial_arr,double Theta) nogil:
	cdef int r
	alpha = get_alpha(itertion_number,n0,last_six,w,EpsilonAvg)
	spatial_dist_denom(n,m,U,itertion_number,n0,last_six,w,EpsilonAvg,alpha,cluster_spatial_arr,cluster_number,Theta)
	for r in xrange(0,cluster_number):
		dist_mat[m][n][r] = eu_distance(data[m][n],cluster_centres[r],cluster_centres.shape[1]) + spatial_distance(r,cluster_spatial_arr,alpha)
	return 

@boundscheck(False)
@wraparound(False)
def update_U(double[:,:,:] U,double[:,:,:] data,int itertion_number,double [:,:] cluster_centres,double[:,:,:] dist_mat,int row,int col,int cluster_number,int channel,double fuzzy_index,double neg_dist,int maxconn,int n0,double [:] last_six,double w,double EpsilonAvg,double [:] cluster_spatial_arr,double Theta):
	

	cdef int i, j
	n0 = get_n0(n0,last_six,EpsilonAvg,itertion_number)
	with nogil,parallel(num_threads=8):
		for i in prange(row,schedule = 'dynamic'):
			for j in prange(col):
				calculate_distance(j,i,itertion_number,cluster_number,U,data,cluster_centres,dist_mat,n0,last_six,w,EpsilonAvg,cluster_spatial_arr,Theta)
	# with nogil,parallel(num_threads=maxconn):
	# 	for i in prange(row,schedule = 'dynamic'):
	# 		for j in prange(col):
	# 			with gil:
	# 				indices = [i for i in xrange(0,cluster_number) if dist_mat[m][n][i] <= neg_dist ]
	# 				for r in xrange(0,cluster_number):
	# 				if indices and r in indices:
	# 					U[m][n][r] = 1
	# 				elif indices and r not in indices:
	# 					U[m][n][r] = 0
	# 				else:
	# 					U[m][n][r] = 1/sum([pow(dist_mat[m][n][r]/dist_mat[m][n][k],float(2/fuzzy_index-1)) for k in xrange(0,cluster_number)])
	for m in xrange(0,row):
		for n in xrange(0,col):
			indices = [i for i in xrange(0,cluster_number) if dist_mat[m][n][i] <= neg_dist ]
			for r in xrange(0,cluster_number):
				if indices and r in indices:
					U[m][n][r] = 1
				elif indices and r not in indices:
					U[m][n][r] = 0
				else:
					U[m][n][r] = 1/sum([pow(dist_mat[m][n][r]/dist_mat[m][n][k],float(2/fuzzy_index-1)) for k in xrange(0,cluster_number)])
	return U,n0

def get_L(U):
	row = len(U)
	col = len(U[0])
	L = [[0 for x in xrange(0,col)] for y in xrange(0,row)]
	L = np.array(L)
	return L


def fuzzy_cluster(image_path,datetime_str):
	MAX = 1000.0
	Epsilon = 0.0002
	EpsilonAvg = 0.001
	Theta = 0.7
	n0 = 40
	last_six = np.zeros((6))
	w = 0.9
	fuzzy_index = 1.3
	fuzzy_index_prev = 1.3
	neg_dist = 0
	cluster_number = 6
	cluster_color = {0:[255,255,0],1:[128,255,0],2:[0,128,255],3:[255,0,255],4:[255,0,0],5:[0,0,0]}
	maxconn=8
	itertion_number = 1
	data = get_data_from_pickle(image_path)
	# print data
	row = len(data)
	col = len(data[0])
	channel = len(data[0][0])
	U = initialise_U(data,cluster_number,MAX)
	cluster_centres = np.zeros((cluster_number,channel))
	dist_mat = np.zeros((row,col,cluster_number))
	v_denom = np.zeros((cluster_number))
	cluster_spatial_arr = np.zeros((cluster_number))
	L = np.zeros((row,col))
	L_new = np.zeros((row,col))
	L = segment_image(U,L,row,col,cluster_number)
	save_image_from_L(L,itertion_number,datetime_str,cluster_color,image_path)
	while True:
		cluster_centres = get_cluster_ref(U,data,cluster_centres,v_denom,fuzzy_index,row,col,cluster_number,channel)
		print cluster_centres,'cluster_centres'
		U,n0 = update_U(U,data,itertion_number,cluster_centres,dist_mat,row,col,cluster_number,channel,fuzzy_index,neg_dist,maxconn,n0,last_six,w,EpsilonAvg,cluster_spatial_arr,Theta)
		print itertion_number,'itertion_number'
		L_new = segment_image(U,L_new,row,col,cluster_number)
		to_end,last_six = end_conditon(L_new,L,row,col,Epsilon,last_six)
		L = np.array(L_new, copy=True)
		itertion_number += 1
		save_image_from_L(L,itertion_number,datetime_str,cluster_color,image_path)
		if to_end:
			break
	L = segment_image(U,L,row,col,cluster_number)
	return L

def run_fuzzy(image_path):
	datetime_str = str(datetime.now()).replace(" ","")
	# create_image(image_path)
	image_name = image_path.split(".")[0]
	start_time = timeit.default_timer()
	save_pickle_from_image(image_path)
	L = fuzzy_cluster(image_path,datetime_str)
	print(timeit.default_timer() - start_time),"total execution time"
	f = open("seg_"+image_name+".pickle","wb")
	pickle.dump(L, f)
	f.close()

