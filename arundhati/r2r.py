import pickle, math

esd={
	1:0.9832,
	15:0.9836,
	32:0.9853,
	46:0.9878,
	60:0.9909,
	74:0.9945,
	91:0.9993,
	106:1.0033,
	121:1.0076,
	135:1.0109,
	152:1.0140,
	166:1.0158,
	182:1.0167,
	196:1.0165,
	213:1.0149,
	227:1.0128,
	242:1.0092,
	258:1.0057,
	274:1.0011,
	288:0.9972,
	305:0.9925,
	319:0.9892,
	335:0.9860,
	349:0.9843,
	365:0.9833
}

f=open('irrad.pickle','r')
irrad=pickle.load(f)
f.close()

def radianceToReflectance(rad,d,esun,theta):
	num=math.pi*rad*d*d
	denom=esun*math.cos(math.radians(theta))
	if denom==0:return 10000000
	return num/denom

def getESD(d):
	d_val=[]
	for d1 in esd:
		d_val.append((abs(d-d1),esd[d1]))
	d_val.sort()
	return d_val[0][1]

def reflectanceVector(v,meta):
	new_v=[]
	centers=meta[0]
	d=getESD(meta[1])
	theta=meta[2]
	for i in range(len(v)):
		new_v.append(radianceToReflectance(v[i],d,irrad[centers[i]],theta))
	return new_v