# ML_KGP
hyperspectral image analysis project - part 2
run the code using the below command

python fuzzy_c_spatial_spectral.py image_file_path

parameters to tune before running the code

1) cluster_number : No of max clusters
2) cluster_color : A dictionary for colors of clusters
3) Epsilon : Halting criteria, difference between 2 iteration
4) fuzzy_index : fuzziness in clustering

All the parameters are available in fuzzy_c_spatial_spectral.py

The output will the created in the same folder as input image 
Ex input image file /a/b/c.jpeg
output /a/b/c_time/c_1.jpeg,/a/b/c_time/c_2.jpeg,/a/b/c_time/c_3.jpeg ...

Requirements osgeo,numpy,PIL,multiprocessing,pickle