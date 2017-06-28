from fuzzy_mat_asyc import segment
import timeit
pickle_data_file="data/hsi_pickle.pickle"
cluster_number = 6
output_path = "output/fuzzy_c_means_async_weight"
start_time = timeit.default_timer()
segment(pickle_data_file,cluster_number,output_path)
print (timeit.default_timer() - start_time),"total execution time"
