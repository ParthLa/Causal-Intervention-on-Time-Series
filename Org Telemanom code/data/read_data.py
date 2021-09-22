import numpy as np
import os

train = 'train/'
test = 'test/'

train_lbl = 'train_lbl.txt'
test_lbl = 'test_lbl.txt'

# A1_train = np.load(train+'A-1.npy')
# A1_test = np.load(test+'A-1.npy')
# # print(A1_train.shape)
# # print(A1_test.shape)
# print(A1_test[4700:4710])
# print("------------------")
# print(A1_test[4600:4610])

def anomaly(vec):
	temp=0
	if(np.sum(vec)<=1):
		temp=1
	# elif(vec[0]!=1):
	# 	temp=1
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
				if(anomaly(vec)==1):
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
				if(anomaly(vec)==1):
					ans[i]=1
			t.write(file)
			t.write('\n')
			t.write(str(ans))
			t.write('\n')
	t.close()
