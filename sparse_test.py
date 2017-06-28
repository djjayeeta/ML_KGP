import pickle
import itertools
import numpy as np
from scipy import sparse
from helper import image_helper as ih
import timeit
from make_beta import weights_HSI
def get_beta(r,row_size,col_size):
        theta = 0.7
        e = 2.71828183
        max_no_of_points = 500
        all_points = np.arange(0,r)
        all_points_row = all_points/col_size
        all_points_col = all_points%col_size
        cols = [[] for i in xrange(0,r)]
        rows = [[] for i in xrange(0,r)]
        values = [[] for i in xrange(0,r)]
        start_time = timeit.default_timer()
        for i in xrange(0,r):
                if i%100 ==0:
                        print i,timeit.default_timer()-start_time
                        start_time = timeit.default_timer()
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
        with open("row_col.pickle","wb") as fp:
                pickle.dump({'rows':rows,'cols':cols,'values':values},fp, protocol=pickle.HIGHEST_PROTOCOL)
        # with open("row_col.pickle","rb") as fp:
        #         picke_data = pickle.load(fp)
        #         rows = picke_data['rows']
        #         cols = picke_data['cols']
        #         values = picke_data['values']
        cols = list(itertools.chain(*cols))
        rows = list(itertools.chain(*rows))
        values = list(itertools.chain(*values))
        beta = sparse.csr_matrix((values,(rows,cols)),shape = (r,r))
        beta_pickle_path = "data/hsi_pickle_beta.pickle"
        with open(beta_pickle_path,"wb") as fp:
            pickle.dump(beta,fp, protocol=pickle.HIGHEST_PROTOCOL)
        return beta

row_size=272
col_size=292
r = row_size*col_size
start_time = timeit.default_timer()
#get_beta(r,row_size,col_size)
pickle_data_file="data/hsi_pickle.pickle"
image = ih.get_pickle_object_as_numpy(pickle_data_file)
beta = weights_HSI(image)
beta_pickle_path = "data/hsi_pickle_beta_new.pickle"
with open(beta_pickle_path,"wb") as fp:
	pickle.dump(beta,fp, protocol=pickle.HIGHEST_PROTOCOL)
print timeit.default_timer()-start_time,"execution time"
