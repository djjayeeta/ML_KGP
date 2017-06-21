import numpy as np
import numpy.matlib
import timeit
from multiprocessing import Pool
maxconn = 8


def threshold_u(x):
	m,n = x.shape
	# m = 6
	s = 1.0 / m
	gs = s / 8
	weight = 5
	y = np.mgrid[0:(0.5+s):s]
	y = y[:-1] if y[-1]>0.5 else y
	dim = m-1
	delta = np.meshgrid(*[y]*dim)
	if dim == 1:
		delta[0] = delta[0].reshape(delta[0].shape[0],1)
	delta.append([])
	temp = delta[0] * 0
	for i in xrange(0,m-1):
		temp = temp + delta[i]
	delta[m-1] = 1 - temp
	idx = np.bitwise_and(delta[m-1]>=0,delta[m-1]<=0.5)
	for i in xrange(0,m):
		delta[i] = delta[i][idx]
	delta = np.array(delta[:])
	ngrid = delta.shape[1]
	fdel = np.zeros((m,ngrid))
	finter = np.zeros(ngrid)
	A = splx_edge_proj_multi(x)
	data_inputs = [[] for i in xrange(0,ngrid)]
	for i in xrange(0,ngrid):
		data_inputs[i] = [A,delta[:,i],m,gs,n,i]
	pool = Pool(maxconn) # on 4 processors
	outputs = pool.map(calculate_fdel_pointwise, data_inputs)
	pool.close()
	pool.join()
	splx_edge_proj_multi_avg_time = 0.0
	fdelta_inner_func_avg_time = 0.0
	rest_avg_time = 0.0
	for i in xrange(0,ngrid):
		l_th_out = outputs[i]
		fdel[:,i] = l_th_out[0]
		finter[i] = l_th_out[1]
		splx_edge_proj_multi_avg_time += l_th_out[2]
		fdelta_inner_func_avg_time += l_th_out[3]
		rest_avg_time += l_th_out[4]	
	print splx_edge_proj_multi_avg_time/ngrid,'splx_edge_proj_multi_avg_time'
	print fdelta_inner_func_avg_time/ngrid,'fdelta_inner_func_avg_time'
	print rest_avg_time/ngrid,'rest_avg_time'
	# for i in xrange(0,ngrid):
	# 	mid = splx_edge_proj_multi(delta[:,i])
	# 	temp,temp_2nd = fdelta_inner_func(A, mid, 0, m, gs, m-1)
	# 	temp2 = np.reshape(temp,(m,n),order='F')
	# 	cluster = np.argmax(temp2,axis=0)
	# 	temp2 = np.bincount(cluster,minlength=m)
	# 	fdel[:, i] = temp2 / float(n)
	# 	finter[i] = 1 - np.sum(temp_2nd,axis=0)/ float(n)

	fobj = -np.log(np.prod(fdel,axis=0))+ weight*np.exp(finter)
	idx = np.argmin(fobj)
	optdelta = delta[:,idx]
	mid = splx_edge_proj_multi(optdelta)
	mid = np.matlib.repmat(mid,n,1)
	thres = np.zeros((n,1))
	temp = (A>=mid)
	temp = (np.sum(temp,axis=1)>=m)
	temp = np.reshape(temp,(m,n),order='F')
	for j in xrange(1,m):
		idx = (np.sum(temp[0:j,:],axis=0)).astype(dtype=np.bool)
		temp[j,idx] = 0
	thres = np.argmax(temp,axis=0)
	outhard = np.zeros((n,m))
	for i in xrange(0,m):
		idx = (thres==i)
		outhard[idx,i] = 1
	return outhard

def calculate_fdel_pointwise(params):
	A,delta_i,m,gs,n,i = params[0],params[1],params[2],params[3],params[4],params[5]
	start_time = timeit.default_timer()
	print i,'calculate_fdel_pointwise'
	mid = splx_edge_proj_multi(delta_i)
	splx_edge_proj_multi_time = (timeit.default_timer() - start_time)
	start_time = timeit.default_timer()
	temp,temp_2nd = fdelta_inner_func(A, mid, 0, m, gs, m-1)
	fdelta_inner_func_time = (timeit.default_timer() - start_time)
	start_time = timeit.default_timer()
	temp2 = np.reshape(temp,(m,n),order='F')
	cluster = np.argmax(temp2,axis=0)
	temp2 = np.bincount(cluster,minlength=m)
	fdel_i = temp2 / float(n)
	finter_i = 1 - np.sum(temp_2nd,axis=0)/ float(n)
	rest_time = (timeit.default_timer() - start_time)
	return [fdel_i,finter_i,splx_edge_proj_multi_time,fdelta_inner_func_time,rest_time]



def fdelta_inner_func(A,mid,thres1,thres2,thres1_2nd,thres2_2nd):
	A_new = A.transpose()
	mid_new = mid.transpose()
	mn = A_new.shape[1]
	m = mid_new.shape[1]
	n = mn / m
	result = np.zeros(mn)
	result_2nd = np.zeros(mn)
	A_pos , mid_pos = 0 , 0

	for j in xrange(0,m):
		mid_pos_temp = mid_pos
		for i in xrange(0,mn):
			A_pos_row,A_pos_col = A_pos/mn,A_pos%mn
			mid_pos_temp_row,mid_pos_temp_col = mid_pos_temp/m,mid_pos_temp%m
			temp = A_new[A_pos_row][A_pos_col] - mid_new[mid_pos_temp_row][mid_pos_temp_col]
			if temp >= thres1:
				result[i] += 1.0
			if temp >= thres1_2nd:
				result_2nd[i] += 1.0
			mid_pos_temp += 1
			if mid_pos_temp == (mid_pos+m):
				mid_pos_temp = mid_pos
			A_pos += 1
		mid_pos += m
	# mn = A.shape[0] # 6r
	# m = mid.shape[0] # 6
	# n = mn / m #r
	# result = np.zeros((mn, 1))
	# result_2nd = np.zeros((mn, 1))
	# A_pos , mid_pos = 0 , 0
	# for j in xrange(0,m):
	# 	mid_pos_temp = mid_pos
	# 	for i in xrange(0,mn):
	# 		A_pos_row,A_pos_col = A_pos%mn,A_pos/mn
	# 		mid_pos_temp_row,mid_pos_temp_col = mid_pos_temp%m,mid_pos_temp/m
	# 		temp = A[A_pos_row][A_pos_col] - mid[mid_pos_temp_row][mid_pos_temp_col]
	# 		if (temp >= thres1):
	# 			result[i] += 1.0
	# 		if (temp >= thres1_2nd):
	# 			result_2nd[i] += 1.0
	# 		mid_pos_temp += 1
	# 		if mid_pos_temp == (mid_pos+m):
	# 			mid_pos_temp = mid_pos
	# 		A_pos += 1
	# 	mid_pos += m
	for i in xrange(0,mn):
		if result[i] >= thres2:
			result[i] = 1.0
		else:
			result[i] = 0.0
		if result_2nd[i] >= thres2_2nd:
			result_2nd[i] = 1.0
		else:
			result_2nd[i] = 0.0
	return result,result_2nd



def splx_edge_proj_multi(x):
	if len(x.shape) == 1:
		x = np.reshape(x,(x.shape[0],1))
	n,m = x.shape
	A = np.zeros((n*m,n))
	for i in xrange(0,n):
		for j in xrange(0,n):
			idx = np.mgrid[i:n*m:n]
			A[idx,j] = .5*(x[i,:]-x[j,:]+1)
	return A
