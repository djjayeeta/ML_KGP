from osgeo import gdal
import numpy as np
import pickle
from PIL import Image


image_path = "IMS1_HYSI_GEO_114_05FEB2009_S1_RADIANCE_07_SPBIN.tif"
dataset = gdal.Open(image_path,gdal.GA_ReadOnly)
band_size = (dataset.RasterCount + 1)/3
col = dataset.RasterXSize
row = dataset.RasterYSize
print band_size,col,row
a = [[[0 for x in range(3)]for y in xrange(col)] for z in xrange(row)]
for i in xrange(1,(dataset.RasterCount + 1)/3+1):
	rband = dataset.GetRasterBand(i).ReadAsArray()
	gband = dataset.GetRasterBand(i+band_size).ReadAsArray()
	yband = dataset.GetRasterBand(i+2*band_size).ReadAsArray() if i!=6 else None
	for m in xrange(0,row):
		for n in xrange(0,col):
			if rband[m][n]>65535:
				print m,n,'rband'
			a[m][n][0] = a[m][n][0] + rband[m][n]
			a[m][n][1] = a[m][n][1] + gband[m][n]
			a[m][n][2] = a[m][n][2] + yband[m][n] if yband is not None else a[m][n][2]


for m in xrange(0,row):
	for n in xrange(0,col):
		a[m][n][0] = a[m][n][0]/6
		a[m][n][1] = a[m][n][1]/6
		a[m][n][2] = a[m][n][2]/6

M = np.array(a,dtype='uint16')
f = open("rgb_IMS1_HYSI_GEO_114_05FEB2009_S1_RADIANCE_07_SPBIN.p","wb")
pickle.dump(M, f)
f.close()
M = M.astype(np.uint8)
im = Image.fromarray(M)
im.save("IMS1_HYSI_GEO_114_05FEB2009_S1_RADIANCE_07_SPBIN.jpeg")
