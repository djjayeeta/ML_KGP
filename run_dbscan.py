from dbscan import segment
pickle_file = "data/test_new_image.pickle"
output_file = "data/dbscan_output"
cluster_number = 4
segment(pickle_file,cluster_number,output_file)