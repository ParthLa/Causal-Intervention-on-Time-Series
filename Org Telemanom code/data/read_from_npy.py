import numpy as np
i=0
train_lbl='train_lbl.npy'
# while (train_lbl):
with open(train_lbl, 'rb') as f:
	while(i<82 and f):
		a=np.load(f)
		i+=1
		print(a.shape)
		print(i)
	# print(i)
	f.close()