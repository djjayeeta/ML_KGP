import ss_fuzzy


pickle_file = "data/Indian_pines.pickle"
cluster_number = 16
output_file = "data/output"

ss_fuzzy.segment(pickle_file,cluster_number,output_file)
