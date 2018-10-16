import numpy as np 
import pandas as pd
import tensorflow as tf
import math
import random
import json
import time

def proccess(path):
	test = pd.read_csv(path+"train.csv", index_col="Id")
	print(test)
	# test.drop('Name',axis=1, inplace=True) # delete Name col
	# test.drop('Ticket',axis=1, inplace=True) # delete ticket col
	# test.fillna(0,inplace=True) # replace Nan in Cabin col with 0 
	# test.loc[test['Cabin']!=0,'Cabin']=1
	# test.loc[test['Cabin']==0,'Cabin0']=0 # replace other sets Cabin col with 1
	# test.loc[test['Cabin']==1,'Cabin0']=1
	# test.loc[test['Cabin']==0,'Cabin1']=1 # replace other sets Cabin col with 1
	# test.loc[test['Cabin']==1,'Cabin1']=0 
	# test.pop('Cabin')
	# test.loc[test['Sex'] =='male','Sex0']=1
	# test.loc[test['Sex'] =='female','Sex0']=0 
	# test.loc[test['Sex'] =='male','Sex1']=0
	# test.loc[test['Sex'] =='female','Sex1']=1
	# test.pop('Sex')
	# test.loc[test['Embarked']=='C','Embarked0']=1 # replace Embarked col with num.
	# test.loc[test['Embarked']=='C','Embarked1']=0 # replace Embarked col with num.
	# test.loc[test['Embarked']=='C','Embarked2']=0 # replace Embarked col with num.
	# test.loc[test['Embarked']=='S','Embarked0']=0 # replace Embarked col with num.
	# test.loc[test['Embarked']=='S','Embarked1']=1 # replace Embarked col with num.
	# test.loc[test['Embarked']=='S','Embarked2']=0 # replace Embarked col with num.
	# test.loc[test['Embarked']=='Q','Embarked0']=0 # replace Embarked col with num.
	# test.loc[test['Embarked']=='Q','Embarked1']=0 # replace Embarked col with num.
	# test.loc[test['Embarked']=='Q','Embarked2']=1 # replace Embarked col with num.
	# test.pop('Embarked')
	# test.loc[test['Pclass']==1,'Pclass0']=1 # replace Embarked col with num.
	# test.loc[test['Pclass']==1,'Pclass1']=0 # replace Embarked col with num.
	# test.loc[test['Pclass']==1,'Pclass2']=0 # replace Embarked col with num.
	# test.loc[test['Pclass']==2,'Pclass0']=0 # replace Embarked col with num.
	# test.loc[test['Pclass']==2,'Pclass1']=1 # replace Embarked col with num.
	# test.loc[test['Pclass']==2,'Pclass2']=0 # replace Embarked col with num.
	# test.loc[test['Pclass']==3,'Pclass0']=0 # replace Embarked col with num.
	# test.loc[test['Pclass']==3,'Pclass1']=0 # replace Embarked col with num.
	# test.loc[test['Pclass']==3,'Pclass2']=1 # replace Embarked col with num.
	# test.pop('Pclass')
	# test.fillna(0,inplace=True)
	# test_x=np.array(test).tolist() # train input in list
	# TestSet=[]
	# for i in range(len(test_x)):
	# 	TestSet.append({'x':test_x[i],'y':[-1]})
	# print(len(TestSet[0]['x']))

if __name__ == '__main__':
	proccess('../input/')