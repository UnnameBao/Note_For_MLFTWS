#coding:utf-8
'''
	@DateTime: 	2018-01-25 16:22:47
	@Version: 	1.0
	@Author: 	Unname_Bao
'''

from nltk.probability import FreqDist
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import numpy as np

def load_user_cmd_new(filename):
	cmd_list = []		#存储所有操作序列
	dist     = []		#存储所有命令
	with open(filename) as f:
		i = 0
		x = []			#存储每个操作序列
		for line in f:
			line = line.strip('\n')		#去掉空行
			dist.append(line)			#添加操作命令
			i+=1
			if i==100:
				cmd_list.append(x)		#每计数100个添加操作序列
				x = []					#然后将操作序列清空
				i = 0
	return cmd_list,list(set(dist))

def get_user_cmd_feature_new(user_cmd_list,dist):
	user_cmd_feature = []
	for cmd_block in user_cmd_list:
		v = [0]*len(dist)				#v为向量，初始全为0
		for i in range(len(dist)):
			if dist[i] in cmd_block:
				v[i] = 1				#一旦使用过某序号的命令，置为1
		user_cmd_feature.append(v)
	return user_cmd_feature

def get_label(filename,index=0):		#读取标签，index+1即用户编号
	x=[]
	with open(filename) as f:
		for line in f:
			line = line.strip('\n')
			x.append(int(line.split()[index]))
	return x

if __name__ == '__main__':
	#读取用户操作序列，并做数据清洗
	user_cmd_list,dist = load_user_cmd_new('../Capter5/MasqueradeDat/User3')
	#将数据特征化
	user_cmd_feature = get_user_cmd_feature_new(user_cmd_list,dist)
	#获得操作序列的标签
	labels = get_label('../Capter5/MasqueradeDat/label.txt',2)
	#label.txt中只有后100个序列的标签，前50个都是正常用户的操作序列
	y = [0]*50 + labels
	neigh   = KNeighborsClassifier(n_neighbors = 3)
	#交叉验证，10次随机取样，n_jobs=-1表示使用全部CPU运行
	print(model_selection.cross_val_score(neigh,user_cmd_feature,y,n_jobs=-1,cv=10))