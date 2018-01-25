#coding:utf-8
'''
	@DateTime: 	2018-01-24 14:36:02
	@Version: 	1.0
	@Author: 	Unname_Bao
'''
from nltk.probability import FreqDist
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import operator

N = 30
def load_user_cmd(filename):
	cmd_list = []
	dist_max = []
	dist_min = []
	dist     = []
	with open(filename) as f:
		i = 0
		x = []
		for line in f:
			line = line.strip('\n')
			x.append(line)
			dist.append(line)
			i+=1
			if i==100:
				cmd_list.append(x)
				x = []
				i = 0
	fdist    = sorted(FreqDist(dist).items(),key = operator.itemgetter(1),reverse = True)
	dist_max = set([item[0] for item in fdist[:50]])
	dist_min = set([item[0] for item in fdist[-50:]])
	return cmd_list,dist_max,dist_min 

def get_user_cmd_feature(user_cmd_list,dist_max,dist_min):
	user_cmd_feature = []
	for cmd_block in user_cmd_list:
		f1 	  = len(set(cmd_block))
		fdist = sorted(FreqDist(cmd_block).items(),key = operator.itemgetter(1),reverse = True)
		f2 	  = [item[0] for item in fdist[:10]]
		f3	  = [item[0] for item in fdist[-10:]]
		f2	  = len(set(f2) & set(dist_max))
		f3    = len(set(f3) & set(dist_min))
		x 	  = [f1,f2,f3]
		user_cmd_feature.append(x)
	return user_cmd_feature

def get_label(filename,index=0):
	x=[]
	with open(filename) as f:
		for line in f:
			line = line.strip('\n')
			x.append(int(line.split()[index]))
	return x

if __name__ == '__main__':
	user_cmd_list,user_cmd_dist_max,user_cmd_dist_min = load_user_cmd('../Capter5/MasqueradeDat/User3')
	user_cmd_feature = get_user_cmd_feature(user_cmd_list,user_cmd_dist_max,user_cmd_dist_min)
	labels = get_label('../Capter5/MasqueradeDat/label.txt',2)
	y = [0]*50 + labels
	x_train = user_cmd_feature[0:N]
	y_train  = y[0:N]
	x_test  = user_cmd_feature[N:150]
	y_test  = y[N:150]
	neigh   = KNeighborsClassifier(n_neighbors = 3)
	neigh.fit(x_train,y_train)
	y_predict = neigh.predict(x_test)
	score = np.mean(y_test == y_predict)*100
	print(score)
