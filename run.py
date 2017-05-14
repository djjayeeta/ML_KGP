import fuzzy_c_spatial_spectral as fuzzy
# import numpy as np
# indata = np.random.rand(40,60)
# outdata = np.zeros(shape=indata.shape, dtype='float64') 
# from numpy.lib import pad
# print("shape before", indata.shape)
# indata = pad(indata, (1, 1), 'reflect', reflect_type='odd')  # allow edge calcs
# print("shape after", indata.shape)

# import asd

fuzzy.run_fuzzy("test_new_cython.jpeg")
# print asd.work()
# print indata,outdata