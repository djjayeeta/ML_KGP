from osgeo import gdal
import pickle

image_path = "IMS1_HYSI_GEO_114_05FEB2009_S1_TOA_REFLECTANCE_07_SPBIN.tif"
img = gdal.Open(image_path,gdal.GA_ReadOnly)

row_num = 272
col_num = 292

pixel = {}

for i in range(1,18):
	band = img.GetRasterBand(i).ReadAsArray()
	for m in range(row_num):
		pixel[m] = {}
		for n in range(col_num):
			#if i == 0:
				pixel[m][n] = []
			#	continue
			#(pixel[m][n]).append(band[m][n])

for i in range(1,18):
	band = img.GetRasterBand(i).ReadAsArray()
	for m in range(row_num):
		#pixel[m] = {}
		for n in range(col_num):
			#if i == 0:
			#	pixel[m][n] = []
			#	continue
			(pixel[m][n]).append(band[m][n]*0.001)

f = open("process_tif.p","wb")
pickle.dump(pixel, f)
f.close()



