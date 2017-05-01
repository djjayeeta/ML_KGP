import os
import pickle

path_to_spectral_library='./ASCII/'

folders=os.listdir(path_to_spectral_library)

sig={}

for folder in folders:
	files=os.listdir(path_to_spectral_library+folder)
	sig[folder]={}
	for filename in files:
		name=filename[:-4]
		f=open(path_to_spectral_library+folder+'/'+filename,'r')
		lines=f.readlines()
		f.close()
		data=[]
		for line in lines[16:]:
			line=line.strip()
			if len(line)==0:
				continue
			words=line.split()
			if len(words)!=3:
				print words
			data.append((float(words[0]),float(words[1]),float(words[2])))
		sig[folder][name]=data

f=open('sig.p','w')
pickle.dump(sig,f)
f.close()
