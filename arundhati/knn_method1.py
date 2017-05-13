import matplotlib
matplotlib.use('Agg')
import pickle
import math
import sys
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal
import operator
from sets import Set
from scipy.stats import norm

def d(X,Y):   
	d = (X - Y) ** 2
	return d

def d_min(ref, wavelength):
	wavelength /= 1000
	minD = None
	minTuple = ()
	na_val = -1.23e+34
	nm_val = 0
	for t in ref:
		#ignore the tuple if reflectance not valid
		if t[1] == na_val or t[1] == nm_val:
			continue
		dist = d(t[0], wavelength)
		if minD == None or dist < minD:
			minD = dist
			minTuple = t
	#print "wavelength: ",wavelength," minTuple: ",minTuple,"\n"
	return minTuple[1]	

def neighbours(ref, band_centers):
#find the vector of reflectance for nearest wavelengths corresponding to the band centers
	v = []
	for i in band_centers:		
		v.append(d_min(ref,i))
	return v	

def sqerror(pixel_reflectance, NN_reflectance):
	err = 0
	na_val = -1.23e+34
	nm_val = 0
#	start = 0
#	end = len(NN_reflectance)
	for i in range(len(NN_reflectance)):
		if NN_reflectance[i] == na_val or NN_reflectance[i] == nm_val:
			continue
		err += (pixel_reflectance[i] - NN_reflectance[i]) ** 2 #/ (pixel_reflectance[i] ** 2 * NN_reflectance[i] ** 2) -->some pixel_reflectance[i] values are all zero for certain pixels
	return err

f = open('sig.p','r')
sig = pickle.load(f)
f.close()

f = open('process_tif.p','rb')
pixels = pickle.load(f)
f.close()
 
centers = [522.2,547.4,572.6,597.8,622.9,648,673.1,698.2,723.2,748.3,773.3,798.3,823.2,848.2,873.1,898,922.9]

likelihood = {}
margin = {}
neighbouring_reflectance = {}
for c in sig:
#	if c != "C":
#		continue
	hist_arr = []
	print "class: ",c
	neighbouring_reflectance[c] = {}
	for t in sig[c]:
		print "material: ",t
		v = neighbours(sig[c][t], centers)	#v is the vector of nearest neighbouring wavelengths for the material t in sig[c]
		neighbouring_reflectance[c][t] = v		
		err_arr = []
		for i in range(272):#img.GetRasterBand(1).ReadAsArray().shape is (272,292)
#			print "i= ",i
			for j in range(292):
#				print "j= ",j
#				pixel_reflectance = r2r.reflectanceVector(img.read_pixel(i,j),meta)
				err_arr.append(sqerror(pixels[i][j],v))
		print "error: ",min(err_arr)
		hist_arr.append(min(err_arr))
	n,bins,patches = plt.hist(hist_arr, 50)#, normed = True, facecolor='green', alpha=0.75)
	mu, sigma = norm.fit(hist_arr)
	margin[c] = bins[1]
	likelihood[c] = (n[0]/len(hist_arr)) + (1/sigma)
	margin[c] = bins[1]
#	print "n = ",n,"\n","bins = ",bins,"\n","patches = ",patches
#	print n[0]," ",bins[1]
	plt.xlabel("min. sq. error")
	plt.ylabel("Frequency")
	plt.grid(True)
	plt.title("Material Class: "+c)
	plt.savefig("Sundarbans:Material_"+c+".jpeg")
	plt.clf()
	plt.cla()
	plt.close()
	
sorted_likelihoods = sorted(likelihood.items(), key = operator.itemgetter(1))
num_classes = 6
#print "sorted_likelihoods: ",sorted_likelihoods

type_ = [[Set() for x in range(292)] for y in range(272)]

#for c in sig:
#	for t in sig[c]:
#for i in range(num_classes-1, -1, -1):
#	nextType = sorted_likelihoods[i][0]
#	comp_margin = margin[nextType]

#	for t in sig[nextType]:
#		v = neighbours(sig[nextType][t], centers)
#	comp_margin = margin[c]
for i in range(272):
	print "i = ",i
	for j in range(292):
		for c in sig:
#			print "now in: ",c
			comp_margin = margin[c]
			cPossible = False
			minDeviation = None
			for t in sig[c]:	
#				v = neighbours(sig[c][t], centers)		
				v = neighbouring_reflectance[c][t]
				deviation = sqerror(pixels[i][j],v)
				if deviation <= comp_margin:
					cPossible = True
					#if minDeviation == None or deviation < minDeviation:
					#	minDeviation = deviation
					type_[i][j].add((nextType,deviation))
#			if cPossible:			
#				type_[i][j].add((c,minDeviation))
	
color_pallete={
		'A - Artificial':(200,200,200),#grey
		'C':(255,0,255),#pink
		'L - Liquids':(0,0,255),#blue
		'M - Minerals':(255,255,0),#yellow
		'S - Soils':(255,0,0),#red
		'V - Vegetation':(0,255,0),#green
		'None':(0,0,0)#black
}

size=(272,292)
mode='RGB'

image=Image.new(mode,size)
for i in range(272):
	for j in range(292):
		#assignClass = 'None'
		#minDeviation = None
		#for k in type_[i][j]:
		#	if minDeviation == None or k[1] < minDeviation:
		#		assignClass = k[0]
		#		minDeviation = k[1]

		#image.putpixel((i,j),color_pallete[assignClass])

#image.save("output2.jpeg")
		if sorted_likelihoods[5][0] in type_[i][j]:
			image.putpixel((i,j),color_pallete[sorted_likelihoods[5][0]])
		elif sorted_likelihoods[4][0] in type_[i][j]:
			image.putpixel((i,j),color_pallete[sorted_likelihoods[4][0]])
		elif sorted_likelihoods[3][0] in type_[i][j]:
			image.putpixel((i,j),color_pallete[sorted_likelihoods[3][0]])
		elif sorted_likelihoods[2][0] in type_[i][j]:
			image.putpixel((i,j),color_pallete[sorted_likelihoods[2][0]])
		elif sorted_likelihoods[1][0] in type_[i][j]:
			image.putpixel((i,j),color_pallete[sorted_likelihoods[1][0]])
		elif sorted_likelihoods[0][0] in type_[i][j]:
			image.putpixel((i,j),color_pallete[sorted_likelihoods[0][0]])

image.save("output1.jpeg")
		 

