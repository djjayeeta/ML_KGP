import ss_fuzzy,mrf,dbscan

algo_func = {"mrf":mrf.segment,"fuzzy_c":ss_fuzzy.segment,"dbscan":dbscan.segment}
algo = "mrf" # mrf/fuzzy_c/db_scan
pickle_file = "data/Indian_pines.pickle"
cluster_number = 16
output_file = "data/output"
algo_func[algo_func](pickle_file,cluster_number,output_file)