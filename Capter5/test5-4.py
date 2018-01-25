#coding:utf-8
'''
	@DateTime: 	2018-01-24 16:25:34
	@Version: 	1.0
	@Author: 	Unname_Bao
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

def load_kdd99(filename):
	x = []
	with open(filename) as f:
		x = [line.strip('\n').split(',') for line in f.readlines()]
	return x

def get_rootkit2andNormal(x):
	v = []
	y = []
	for x1 in x:
		if (x1[41] in ['rootkit.','normal.']) and (x1[2] == 'telnet'):
			if x1[41] == 'rootkit':
				y.append(1)
			else:
				y.append(0)
			v.append(x1[9:21])
	w = []
	for x1 in v:
		v1 = []
		for x2 in x1:
			v1.append(float(x2))
		w.append(v1)
	w2 = [[float(x2) for x2 in x1] for x1 in v]
	print(w == w2)
	return w,y

if __name__ == '__main__':
	v = load_kdd99('corrected')
	w,y = get_rootkit2andNormal(v)
	clf = KNeighborsClassifier(n_neighbors = 3)
	print(model_selection.cross_val_score(clf,w,y,n_jobs=-1,cv=10))
