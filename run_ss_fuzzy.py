import ss_fuzzy


pickle_file = "data/test_image.pickle"
cluster_number = 4
output_file = "data/output"

ss_fuzzy.segment(pickle_file,cluster_number,output_file)