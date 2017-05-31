# ML_KGP
Run the commands in requirements.txt to install the requirements

To cythonize run the command

python setup_seg.py build_ext --inplace

This will generate mrf.so,mrf.c,ss_fuzzy.so,ss_fuzzy.c

Now to run segmentation on a image you need a .pickle of the image which will have a 3 dimensional numpy array. the 3rd dimension denoting the spectral dimension of the image.

To run segmentaion change the following parameters in run_seg.py

1) algo (can be fuzzy_c/mrf/db_scan)
2) pickle_file (.pickle file for the input image)
3) cluster_number (no of classes , will be used in fuzzy_c and mrf )
4) output_file (file to store the output, in case of mrf and fuzzy_c outputs in form of .jpeg and .pickle will be stored for every iteration,for dbscan final output will be stored)

Now run the following command
python run_seg.py