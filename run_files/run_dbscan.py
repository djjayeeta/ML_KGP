from dbscan import segment
pickle_file = "data/hsi_pickle.pickle"
output_file = "data/dbscan_output"
cluster_number = 4
segment(pickle_file,cluster_number,output_file)
