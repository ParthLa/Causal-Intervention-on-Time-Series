import numpy as np
import os

train = 'train/'
test = 'test/'

train_lbl = 'train_lbl.txt'
test_lbl = 'test_lbl.txt'

# A1_train = np.load(train+'P-2.npy')
# A1_test = np.load(test+'P-2.npy')
# print(A1_train.shape)
# print(A1_test.shape)
# print(A1_test[0:10]) # normal
# print("------------------")
# print(A1_test[6000:6010]) # anomaly
# print("------------------")
# # print(A1_test[4600:4610]) # normal

def anomaly(vec, avg, delta):
	temp=0
	if (abs(vec[0]-avg)>delta):
		temp=1
	return temp

with open(train_lbl, 'w') as t:
	for file in os.listdir(train):
		# print(file)

		if(file[-3:]=='npy'):
			f_train = np.load(train+file)
			l = f_train.shape[0]
			ans = np.zeros(l)
			for i in range(l):
				vec = f_train[i]
				avg = 0
				length = 3
				lb = max(0, i-length)
				for j in range(lb, i):
					avg+=f_train[j][0]
				if(i!=0):
					avg = avg/(i - lb) 
				delta = 0.5
				if(anomaly(vec, avg, delta)==1):
					ans[i]=1
			t.write(file)
			t.write('\n')
			t.write(str(ans))
			t.write('\n')
	t.close()

with open(test_lbl, 'w') as t:
	for file in os.listdir(test):
		# print(file)

		if(file[-3:]=='npy'):
			f_test = np.load(test+file)
			l = f_test.shape[0]
			ans = np.zeros(l)
			for i in range(l):
				vec = f_test[i]
				avg = 0
				length = 3
				lb = max(0, i-length)
				for j in range(lb, i):
					avg+=f_test[j][0]
				if(i!=0):
					avg = avg/(i - lb) 
				delta = 0.5
				if(anomaly(vec, avg, delta)==1):
					ans[i]=1
			t.write(file)
			t.write('\n')
			t.write(str(ans))
			t.write('\n')
	t.close()
