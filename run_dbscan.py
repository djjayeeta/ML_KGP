from dbscan import kmeans_segment
pickle_file = "Indian_pines.pickle"
output_file = "data/kmeans_output_Indian_pines"
cluster_number = 16
kmeans_segment(pickle_file,cluster_number,output_file)
