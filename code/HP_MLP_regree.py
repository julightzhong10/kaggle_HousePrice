import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import math
import random
import json
import time

def main():

	'''Management Data'''
	data_path='../input/'
	model_folder='../model/MLP/'

	train_BS=20 # batch size for train

	dropout_rate=0.5
	one_epoch=int(1460/train_BS)
	total_step=one_epoch*500 # 5000 epochs
	lrdecay_bias=one_epoch*4 #test performance every 50 epochs by using whole Test set
	lrdecay_rate=0.98
	learning_rate=1e-1
	momentum_rate=0.9
	layerNum=3
	layer=[1048,1048,1048]
	'''Management Data'''

	'''Data prepare, rotate or shuffle'''
	TrainSet,ValSet,TestSet=data_preparer(data_path,1.0)

	print('Data load finished')
	print('Data shuffle finished')
	print('Data prepare finished')
	'''Data prepare, rotate or shuffle'''
	'''Placeholder Data '''
	x=tf.placeholder(tf.float32, [None, 40])
	y_=tf.placeholder(tf.float32, [None,1])
	ist = tf.placeholder(tf.bool,name='ist')
	lr=tf.placeholder(tf.float32)
	'''Placeholder Data'''

	'''output '''
	y,keep_prob=MLP(x,layerNum,layer,ist)
	result=y
	'''output '''

	'''loss & train'''
	final_loss=tf.losses.mean_squared_error(labels=y_, predictions=y)
	#final_loss=tf.losses.hinge_loss(labels=y_, logits=y)
	#inal_loss=tf.losses.absolute_difference(labels=y_, predictions=y)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum_rate).minimize(final_loss)
		#train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(final_loss)
	'''loss & train'''
	'''dash data'''
	#final_prob=tf.reduce_mean(tf.cast(tf.equal(tf.round(y), y_), tf.float32))
	RMSLE=tf.reduce_mean(tf.square(tf.subtract(tf.log(tf.add(tf.abs(y_),1)),tf.log(tf.add(tf.abs(y),1)))))
	'''dash data'''

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('Training Start! ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		for i in range(total_step): # training begin
			timeStart=time.time()
			if(i+1)%lrdecay_bias==0:
				print("epochs:",int((i+1)/lrdecay_bias))
				finalLoss=0
				finalRMSLE=0
				for vi in range(len(ValSet[0])):
					val_batch=next_feed(1,ValSet) 
					eval_loss,eval_RMSLE=sess.run(fetches=[final_loss,RMSLE],feed_dict={x:val_batch[0],y_:val_batch[1],keep_prob:1.0,ist:False})
					finalLoss=finalLoss+eval_loss
					finalRMSLE=finalRMSLE+eval_RMSLE
				print("Val MSE",finalLoss/len(ValSet))
				print("Val RMSLE",math.sqrt(finalRMSLE/len(ValSet[0])))
				learning_rate=learning_rate*lrdecay_rate
				print('lr:',learning_rate)
				print('----------------------------------------')
				
			train_batch=next_feed(train_BS,TrainSet)
			train_step.run(feed_dict={x:train_batch[0],y_:train_batch[1],keep_prob:dropout_rate,ist:True,lr:learning_rate})

		'''for validation'''

		data_Id=TestSet[1]
		data_label=[]
		for i in range(len(TestSet[0])):
			test_batch=next_feed(1,TestSet)
			r=result.eval(feed_dict={x:test_batch[0],keep_prob:1.0,ist:False})
			data_label.append(int(r[0][0]))
		data_result={'Id':data_Id,'SalePrice':data_label}
		df = pd.DataFrame(data_result, columns= ['Id', 'SalePrice'])
		export_csv = df.to_csv (model_folder+'result.csv', index = None, header=True)



		sess.close()
		print('Done',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def MLP(input_x,layers_num,layers,ist):
  keep_prob_MLP = tf.placeholder(tf.float32)# dropout rate

  W_MLP = weight_variable([40, layers[0]],'MLP_layer0')
  h_MLP = tf.nn.dropout(af2(batch_norm(tf.matmul(input_x, W_MLP),ist)),keep_prob_MLP)

  for i in range(1,layers_num):
  	W_MLP = weight_variable([layers[i-1], layers[i]],'MLP_layer'+str(i))
  	h_MLP = tf.nn.dropout(af2(batch_norm(tf.matmul(h_MLP, W_MLP),ist)),keep_prob_MLP)
  
  W_mark = weight_variable([layers[-1], 1],'mark_layer')
  final_mark = batch_norm(tf.matmul(h_MLP, W_mark),ist)

  return final_mark,keep_prob_MLP

def batch_norm(x,ist):
  '''Batch Norm'''
  return tf.contrib.layers.batch_norm(inputs=x,is_training=ist,fused=True,data_format='NHWC',epsilon=1e-3,scale=True,decay=0.9)


def conv2d(x, W,s=[1, 1, 1, 1],p='SAME'):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(input=x,filter=W, strides=s, padding=p,data_format='NHWC')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape,name):
  """weight_variable generates a weight variable of a given shape."""
  return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),regularizer=tf.contrib.layers.l2_regularizer(5e-2))


def af1(x):
  '''change setting for different activation function'''
  return tf.nn.elu(x)
def af2(x):
  '''change setting for different activation function'''
  return tf.nn.tanh(x)


def next_feed(num,dataArr):
  '''pop the next feed in batch size number'''
  iArr=[]
  oArr=[]
  for i in range(num):
    x=dataArr[0].pop(0)
    y=dataArr[1].pop(0)
    dataArr[0].append(x)
    dataArr[1].append(y)
    iArr.append(x)
    oArr.append([y])
  return [iArr,oArr]


def validation_accu(x,y):
	true=0
	for i in range(len(x)):
		true=true+(math.log(x[i]+1)-math.log(y[i]+1))**2
	return true/len(x)

def data_wash_train(train):
	train.fillna(-1,inplace=True)
	#print(np.array(train).tolist()[0])
	train.loc[train['MSZoning']=='A','MSZoning']=0
	train.loc[train['MSZoning']=='C','MSZoning']=1
	train.loc[train['MSZoning']=='C (all)','MSZoning']=1
	train.loc[train['MSZoning']=='FV','MSZoning']=2
	train.loc[train['MSZoning']=='I','MSZoning']=3
	train.loc[train['MSZoning']=='RH','MSZoning']=4
	train.loc[train['MSZoning']=='RL','MSZoning']=5
	train.loc[train['MSZoning']=='RP','MSZoning']=6
	train.loc[train['MSZoning']=='RM','MSZoning']=7

	train.loc[train['Street']=='Grvl','Street']=0
	train.loc[train['Street']=='Pave','Street']=1

	train.loc[train['Alley']=='Grvl','Alley']=0
	train.loc[train['Alley']=='Pave','Alley']=1
	#train.loc[train['Alley']=='NA','Alley']=2

	train.loc[train['LotShape']=='Reg','LotShape']=0
	train.loc[train['LotShape']=='IR1','LotShape']=1
	train.loc[train['LotShape']=='IR2','LotShape']=2
	train.loc[train['LotShape']=='IR3','LotShape']=3


	train.loc[train['LandContour']=='Lvl','LandContour']=0
	train.loc[train['LandContour']=='Bnk','LandContour']=1
	train.loc[train['LandContour']=='HLS','LandContour']=2
	train.loc[train['LandContour']=='Low','LandContour']=3

	train.loc[train['Utilities']=='AllPub','Utilities']=0
	train.loc[train['Utilities']=='NoSewr','Utilities']=1
	train.loc[train['Utilities']=='NoSeWa','Utilities']=2
	#train.loc[train['Utilities']=='ELO','Utilities']=3


	train.loc[train['LotConfig']=='Inside','LotConfig']=0
	train.loc[train['LotConfig']=='Corner','LotConfig']=1
	train.loc[train['LotConfig']=='CulDSac','LotConfig']=2
	train.loc[train['LotConfig']=='FR2','LotConfig']=3
	train.loc[train['LotConfig']=='FR3','LotConfig']=4

	train.loc[train['LandSlope']=='Gtl','LandSlope']=0
	train.loc[train['LandSlope']=='Mod','LandSlope']=1
	train.loc[train['LandSlope']=='Sev','LandSlope']=2

	train.loc[train['Neighborhood']=='Blmngtn','Neighborhood']=0
	train.loc[train['Neighborhood']=='Blueste','Neighborhood']=1
	train.loc[train['Neighborhood']=='BrDale','Neighborhood']=2
	train.loc[train['Neighborhood']=='ClearCr','Neighborhood']=3
	train.loc[train['Neighborhood']=='CollgCr','Neighborhood']=4
	train.loc[train['Neighborhood']=='Crawfor','Neighborhood']=5
	train.loc[train['Neighborhood']=='Edwards','Neighborhood']=6
	train.loc[train['Neighborhood']=='Gilbert','Neighborhood']=7
	train.loc[train['Neighborhood']=='IDOTRR','Neighborhood']=8
	train.loc[train['Neighborhood']=='MeadowV','Neighborhood']=9
	train.loc[train['Neighborhood']=='Mitchel','Neighborhood']=10
	train.loc[train['Neighborhood']=='Names','Neighborhood']=11
	train.loc[train['Neighborhood']=='NAmes','Neighborhood']=11
	train.loc[train['Neighborhood']=='NoRidge','Neighborhood']=12
	train.loc[train['Neighborhood']=='NPkVill','Neighborhood']=13
	train.loc[train['Neighborhood']=='NridgHt','Neighborhood']=14
	train.loc[train['Neighborhood']=='NWAmes','Neighborhood']=15
	train.loc[train['Neighborhood']=='OldTown','Neighborhood']=16
	train.loc[train['Neighborhood']=='SWISU','Neighborhood']=17
	train.loc[train['Neighborhood']=='Sawyer','Neighborhood']=18
	train.loc[train['Neighborhood']=='SawyerW','Neighborhood']=19
	train.loc[train['Neighborhood']=='Somerst','Neighborhood']=20
	train.loc[train['Neighborhood']=='StoneBr','Neighborhood']=21
	train.loc[train['Neighborhood']=='Timber','Neighborhood']=22	
	train.loc[train['Neighborhood']=='Veenker','Neighborhood']=23
	train.loc[train['Neighborhood']=='BrkSide','Neighborhood']=23

	train.loc[train['Condition1']=='Artery','Condition1']=0
	train.loc[train['Condition1']=='Feedr','Condition1']=1
	train.loc[train['Condition1']=='Norm','Condition1']=2
	train.loc[train['Condition1']=='RRNn','Condition1']=3
	train.loc[train['Condition1']=='RRAn','Condition1']=4
	train.loc[train['Condition1']=='PosN','Condition1']=5
	train.loc[train['Condition1']=='PosA','Condition1']=6
	train.loc[train['Condition1']=='RRNe','Condition1']=7
	train.loc[train['Condition1']=='RRAe','Condition1']=8

	train.loc[train['Condition2']=='Artery','Condition2']=0
	train.loc[train['Condition2']=='Feedr','Condition2']=0
	train.loc[train['Condition2']=='Norm','Condition2']=0
	train.loc[train['Condition2']=='RRNn','Condition2']=0
	train.loc[train['Condition2']=='RRAn','Condition2']=0
	train.loc[train['Condition2']=='PosN','Condition2']=0
	train.loc[train['Condition2']=='PosA','Condition2']=0
	train.loc[train['Condition2']=='RRNe','Condition2']=0
	train.loc[train['Condition2']=='RRAe','Condition2']=0
	
	train.loc[train['BldgType']=='1Fam','BldgType']=0
	train.loc[train['BldgType']=='2FmCon','BldgType']=1
	train.loc[train['BldgType']=='2fmCon','BldgType']=1
	train.loc[train['BldgType']=='Duplex','BldgType']=2
	train.loc[train['BldgType']=='TwnhsE','BldgType']=3
	train.loc[train['BldgType']=='Twnhs','BldgType']=3
	#train.loc[train['BldgType']=='TwnhsI','BldgType']=4

	train.loc[train['HouseStyle']=='1Story','HouseStyle']=0
	train.loc[train['HouseStyle']=='1.5Fin','HouseStyle']=1
	train.loc[train['HouseStyle']=='1.5Unf','HouseStyle']=2
	train.loc[train['HouseStyle']=='2Story','HouseStyle']=3
	train.loc[train['HouseStyle']=='2.5Fin','HouseStyle']=4
	train.loc[train['HouseStyle']=='2.5Unf','HouseStyle']=5
	train.loc[train['HouseStyle']=='SFoyer','HouseStyle']=6		
	train.loc[train['HouseStyle']=='SLvl','HouseStyle']=7

	train.loc[train['RoofStyle']=='Flat','RoofStyle']=0
	train.loc[train['RoofStyle']=='Gable','RoofStyle']=1
	train.loc[train['RoofStyle']=='Gambrel','RoofStyle']=2
	train.loc[train['RoofStyle']=='Hip','RoofStyle']=3
	train.loc[train['RoofStyle']=='Mansard','RoofStyle']=4
	train.loc[train['RoofStyle']=='Shed','RoofStyle']=5

	train.loc[train['RoofMatl']=='ClyTile','RoofMatl']=0
	train.loc[train['RoofMatl']=='CompShg','RoofMatl']=1
	train.loc[train['RoofMatl']=='Membran','RoofMatl']=2
	train.loc[train['RoofMatl']=='Metal','RoofMatl']=3
	train.loc[train['RoofMatl']=='Roll','RoofMatl']=4
	train.loc[train['RoofMatl']=='Tar&Grv','RoofMatl']=5
	train.loc[train['RoofMatl']=='WdShake','RoofMatl']=6	
	train.loc[train['RoofMatl']=='WdShngl','RoofMatl']=7

	train.loc[train['Exterior1st']=='AsbShng','Exterior1st']=0
	train.loc[train['Exterior1st']=='AsphShn','Exterior1st']=1
	train.loc[train['Exterior1st']=='BrkComm','Exterior1st']=2
	train.loc[train['Exterior1st']=='Brk Cmn','Exterior1st']=2
	train.loc[train['Exterior1st']=='BrkFace','Exterior1st']=3
	train.loc[train['Exterior1st']=='CBlock','Exterior1st']=4
	train.loc[train['Exterior1st']=='HdBoard','Exterior1st']=5	
	train.loc[train['Exterior1st']=='ImStucc','Exterior1st']=6
	train.loc[train['Exterior1st']=='MetalSd','Exterior1st']=7
	train.loc[train['Exterior1st']=='Other','Exterior1st']=8
	train.loc[train['Exterior1st']=='Plywood','Exterior1st']=9
	train.loc[train['Exterior1st']=='PreCast','Exterior1st']=10
	train.loc[train['Exterior1st']=='Stone','Exterior1st']=11
	train.loc[train['Exterior1st']=='Stucco','Exterior1st']=12
	train.loc[train['Exterior1st']=='VinylSd','Exterior1st']=13
	train.loc[train['Exterior1st']=='Wd Sdng','Exterior1st']=14
	train.loc[train['Exterior1st']=='WdShing','Exterior1st']=15
	train.loc[train['Exterior1st']=='CemntBd','Exterior1st']=16
	#train.loc[train['Exterior1st']=='Wd Shng','Exterior1st']=17

	train.loc[train['Exterior2nd']=='AsbShng','Exterior2nd']=0
	train.loc[train['Exterior2nd']=='AsphShn','Exterior2nd']=1
	train.loc[train['Exterior2nd']=='BrkComm','Exterior2nd']=2
	train.loc[train['Exterior2nd']=='Brk Cmn','Exterior2nd']=2
	train.loc[train['Exterior2nd']=='BrkFace','Exterior2nd']=3
	train.loc[train['Exterior2nd']=='CBlock','Exterior2nd']=4
	train.loc[train['Exterior2nd']=='HdBoard','Exterior2nd']=5	
	train.loc[train['Exterior2nd']=='ImStucc','Exterior2nd']=6
	train.loc[train['Exterior2nd']=='MetalSd','Exterior2nd']=7
	train.loc[train['Exterior2nd']=='Other','Exterior2nd']=8
	train.loc[train['Exterior2nd']=='Plywood','Exterior2nd']=9
	train.loc[train['Exterior2nd']=='PreCast','Exterior2nd']=10
	train.loc[train['Exterior2nd']=='Stone','Exterior2nd']=11
	train.loc[train['Exterior2nd']=='Stucco','Exterior2nd']=12
	train.loc[train['Exterior2nd']=='VinylSd','Exterior2nd']=13
	train.loc[train['Exterior2nd']=='Wd Sdng','Exterior2nd']=14
	train.loc[train['Exterior2nd']=='WdShing','Exterior2nd']=15
	train.loc[train['Exterior2nd']=='CmentBd','Exterior2nd']=16
	train.loc[train['Exterior2nd']=='Wd Shng','Exterior2nd']=17

	train.loc[train['MasVnrType']=='BrkCmn','MasVnrType']=0
	train.loc[train['MasVnrType']=='BrkFace','MasVnrType']=1
	train.loc[train['MasVnrType']=='CBlock','MasVnrType']=2
	train.loc[train['MasVnrType']=='None','MasVnrType']=3
	train.loc[train['MasVnrType']=='Stone','MasVnrType']=4

	train.loc[train['ExterQual']=='Ex','ExterQual']=0
	train.loc[train['ExterQual']=='Gd','ExterQual']=1
	train.loc[train['ExterQual']=='TA','ExterQual']=2
	train.loc[train['ExterQual']=='Fa','ExterQual']=3
	#train.loc[train['ExterQual']=='Po','ExterQual']=4

	train.loc[train['ExterCond']=='Ex','ExterCond']=0
	train.loc[train['ExterCond']=='Gd','ExterCond']=1
	train.loc[train['ExterCond']=='TA','ExterCond']=2
	train.loc[train['ExterCond']=='Fa','ExterCond']=3
	train.loc[train['ExterCond']=='Po','ExterCond']=4

	train.loc[train['Foundation']=='BrkTil','Foundation']=0
	train.loc[train['Foundation']=='CBlock','Foundation']=1
	train.loc[train['Foundation']=='PConc','Foundation']=2
	train.loc[train['Foundation']=='Slab','Foundation']=3
	train.loc[train['Foundation']=='Stone','Foundation']=4
	train.loc[train['Foundation']=='Wood','Foundation']=5

	train.loc[train['BsmtQual']=='Ex','BsmtQual']=0
	train.loc[train['BsmtQual']=='Gd','BsmtQual']=1
	train.loc[train['BsmtQual']=='TA','BsmtQual']=2
	train.loc[train['BsmtQual']=='Fa','BsmtQual']=3
	#train.loc[train['BsmtQual']=='Po','BsmtQual']=4
	#train.loc[train['BsmtQual']=='NA','BsmtQual']=5
	
	train.loc[train['BsmtCond']=='Ex','BsmtCond']=0
	train.loc[train['BsmtCond']=='Gd','BsmtCond']=1
	train.loc[train['BsmtCond']=='TA','BsmtCond']=2
	train.loc[train['BsmtCond']=='Fa','BsmtCond']=3
	train.loc[train['BsmtCond']=='Po','BsmtCond']=4
	#train.loc[train['BsmtCond']=='NA','BsmtCond']=5

	train.loc[train['BsmtExposure']=='Gd','BsmtExposure']=0
	train.loc[train['BsmtExposure']=='Av','BsmtExposure']=1
	train.loc[train['BsmtExposure']=='Mn','BsmtExposure']=2
	train.loc[train['BsmtExposure']=='No','BsmtExposure']=3
	#train.loc[train['BsmtExposure']=='NA','BsmtExposure']=4

	train.loc[train['BsmtFinType1']=='GLQ','BsmtFinType1']=0
	train.loc[train['BsmtFinType1']=='ALQ','BsmtFinType1']=1
	train.loc[train['BsmtFinType1']=='BLQ','BsmtFinType1']=2
	train.loc[train['BsmtFinType1']=='LwQ','BsmtFinType1']=3
	train.loc[train['BsmtFinType1']=='Unf','BsmtFinType1']=4
	train.loc[train['BsmtFinType1']=='Rec','BsmtFinType1']=5
	#train.loc[train['BsmtFinType1']=='NA','BsmtFinType1']=5

	train.loc[train['BsmtFinType2']=='GLQ','BsmtFinType2']=0
	train.loc[train['BsmtFinType2']=='ALQ','BsmtFinType2']=1
	train.loc[train['BsmtFinType2']=='BLQ','BsmtFinType2']=2
	train.loc[train['BsmtFinType2']=='LwQ','BsmtFinType2']=3
	train.loc[train['BsmtFinType2']=='Unf','BsmtFinType2']=4
	train.loc[train['BsmtFinType2']=='Rec','BsmtFinType2']=5
	#train.loc[train['BsmtFinType2']=='NA','BsmtFinType2']=5

	train.loc[train['Heating']=='Floor','Heating']=0
	train.loc[train['Heating']=='GasA','Heating']=1
	train.loc[train['Heating']=='GasW','Heating']=2
	train.loc[train['Heating']=='Grav','Heating']=3
	train.loc[train['Heating']=='OthW','Heating']=4
	train.loc[train['Heating']=='Wall','Heating']=5

	train.loc[train['HeatingQC']=='Ex','HeatingQC']=0
	train.loc[train['HeatingQC']=='Gd','HeatingQC']=1
	train.loc[train['HeatingQC']=='TA','HeatingQC']=2
	train.loc[train['HeatingQC']=='Fa','HeatingQC']=3
	train.loc[train['HeatingQC']=='Po','HeatingQC']=4

	train.loc[train['CentralAir']=='N','CentralAir']=0
	train.loc[train['CentralAir']=='Y','CentralAir']=1

	train.loc[train['Electrical']=='SBrkr','Electrical']=0
	train.loc[train['Electrical']=='FuseA','Electrical']=1
	train.loc[train['Electrical']=='FuseF','Electrical']=2
	train.loc[train['Electrical']=='FuseP','Electrical']=3
	train.loc[train['Electrical']=='Mix','Electrical']=4

	train.loc[train['KitchenQual']=='Ex','KitchenQual']=0
	train.loc[train['KitchenQual']=='Gd','KitchenQual']=1
	train.loc[train['KitchenQual']=='TA','KitchenQual']=2
	train.loc[train['KitchenQual']=='Fa','KitchenQual']=3
	#train.loc[train['KitchenQual']=='Po','KitchenQual']=4

	train.loc[train['Functional']=='Typ','Functional']=0
	train.loc[train['Functional']=='Min1','Functional']=1
	train.loc[train['Functional']=='Min2','Functional']=2
	train.loc[train['Functional']=='Mod','Functional']=3
	train.loc[train['Functional']=='Maj1','Functional']=4
	train.loc[train['Functional']=='Maj2','Functional']=5
	train.loc[train['Functional']=='Sev','Functional']=6
	#train.loc[train['Functional']=='Sal','Functional']=7

	train.loc[train['FireplaceQu']=='Ex','FireplaceQu']=0
	train.loc[train['FireplaceQu']=='Gd','FireplaceQu']=1
	train.loc[train['FireplaceQu']=='TA','FireplaceQu']=2
	train.loc[train['FireplaceQu']=='Fa','FireplaceQu']=3
	train.loc[train['FireplaceQu']=='Po','FireplaceQu']=4
	#train.loc[train['FireplaceQu']=='NA','FireplaceQu']=5

	train.loc[train['GarageType']=='2Types','GarageType']=0
	train.loc[train['GarageType']=='Attchd','GarageType']=1
	train.loc[train['GarageType']=='Basment','GarageType']=2
	train.loc[train['GarageType']=='BuiltIn','GarageType']=3
	train.loc[train['GarageType']=='CarPort','GarageType']=4
	train.loc[train['GarageType']=='Detchd','GarageType']=5
	#train.loc[train['GarageType']=='NA','GarageType']=6

	train.loc[train['GarageFinish']=='Fin','GarageFinish']=0
	train.loc[train['GarageFinish']=='RFn','GarageFinish']=1
	train.loc[train['GarageFinish']=='Unf','GarageFinish']=3
	#train.loc[train['GarageFinish']=='NA','GarageFinish']=4

	train.loc[train['GarageQual']=='Ex','GarageQual']=0
	train.loc[train['GarageQual']=='Gd','GarageQual']=1
	train.loc[train['GarageQual']=='TA','GarageQual']=2
	train.loc[train['GarageQual']=='Fa','GarageQual']=3
	train.loc[train['GarageQual']=='Po','GarageQual']=4
	#train.loc[train['GarageQual']=='NA','GarageQual']=5

	train.loc[train['GarageCond']=='Ex','GarageCond']=0
	train.loc[train['GarageCond']=='Gd','GarageCond']=1
	train.loc[train['GarageCond']=='TA','GarageCond']=2
	train.loc[train['GarageCond']=='Fa','GarageCond']=3
	train.loc[train['GarageCond']=='Po','GarageCond']=4
	#train.loc[train['GarageCond']=='NA','GarageCond']=5

	train.loc[train['PavedDrive']=='Y','PavedDrive']=0
	train.loc[train['PavedDrive']=='P','PavedDrive']=1
	train.loc[train['PavedDrive']=='N','PavedDrive']=2

	train.loc[train['PoolQC']=='Ex','PoolQC']=0
	train.loc[train['PoolQC']=='Gd','PoolQC']=1
	train.loc[train['PoolQC']=='TA','PoolQC']=2
	train.loc[train['PoolQC']=='Fa','PoolQC']=3
	#train.loc[train['PoolQC']=='NA','PoolQC']=4

	train.loc[train['Fence']=='GdPrv','Fence']=0
	train.loc[train['Fence']=='MnPrv','Fence']=1
	train.loc[train['Fence']=='GdWo','Fence']=2
	train.loc[train['Fence']=='MnWw','Fence']=3
	#train.loc[train['Fence']=='NA','Fence']=4

	train.loc[train['MiscFeature']=='Elev','MiscFeature']=0
	train.loc[train['MiscFeature']=='Gar2','MiscFeature']=1
	train.loc[train['MiscFeature']=='Othr','MiscFeature']=2
	train.loc[train['MiscFeature']=='Shed','MiscFeature']=3
	train.loc[train['MiscFeature']=='TenC','MiscFeature']=4
	#train.loc[train['MiscFeature']=='NA','MiscFeature']=5

	train.loc[train['SaleType']=='WD','SaleType']=0
	train.loc[train['SaleType']=='CWD','SaleType']=1
	train.loc[train['SaleType']=='VWD','SaleType']=2
	train.loc[train['SaleType']=='New','SaleType']=3
	train.loc[train['SaleType']=='COD','SaleType']=4
	train.loc[train['SaleType']=='Con','SaleType']=5
	train.loc[train['SaleType']=='ConLw','SaleType']=6
	train.loc[train['SaleType']=='ConLI','SaleType']=7
	train.loc[train['SaleType']=='ConLD','SaleType']=8
	train.loc[train['SaleType']=='Oth','SaleType']=9

	train.loc[train['SaleCondition']=='Normal','SaleCondition']=9
	train.loc[train['SaleCondition']=='Abnorml','SaleCondition']=9
	train.loc[train['SaleCondition']=='AdjLand','SaleCondition']=9
	train.loc[train['SaleCondition']=='Alloca','SaleCondition']=9
	train.loc[train['SaleCondition']=='Family','SaleCondition']=9
	train.loc[train['SaleCondition']=='Partial','SaleCondition']=9

	return train

def data_wash_test(train):
	train.fillna(-1,inplace=True)
	#print(np.array(train).tolist()[0])
	train.loc[train['MSZoning']=='A','MSZoning']=0
	train.loc[train['MSZoning']=='C','MSZoning']=1
	train.loc[train['MSZoning']=='C (all)','MSZoning']=1
	train.loc[train['MSZoning']=='FV','MSZoning']=2
	train.loc[train['MSZoning']=='I','MSZoning']=3
	train.loc[train['MSZoning']=='RH','MSZoning']=4
	train.loc[train['MSZoning']=='RL','MSZoning']=5
	train.loc[train['MSZoning']=='RP','MSZoning']=6
	train.loc[train['MSZoning']=='RM','MSZoning']=7

	train.loc[train['Street']=='Grvl','Street']=0
	train.loc[train['Street']=='Pave','Street']=1

	train.loc[train['Alley']=='Grvl','Alley']=0
	train.loc[train['Alley']=='Pave','Alley']=1
	#train.loc[train['Alley']=='NA','Alley']=2

	train.loc[train['LotShape']=='Reg','LotShape']=0
	train.loc[train['LotShape']=='IR1','LotShape']=1
	train.loc[train['LotShape']=='IR2','LotShape']=2
	train.loc[train['LotShape']=='IR3','LotShape']=3


	train.loc[train['LandContour']=='Lvl','LandContour']=0
	train.loc[train['LandContour']=='Bnk','LandContour']=1
	train.loc[train['LandContour']=='HLS','LandContour']=2
	train.loc[train['LandContour']=='Low','LandContour']=3

	train.loc[train['Utilities']=='AllPub','Utilities']=0
	#train.loc[train['Utilities']=='NoSewr','Utilities']=1
	#train.loc[train['Utilities']=='NoSeWa','Utilities']=2
	#train.loc[train['Utilities']=='ELO','Utilities']=3


	train.loc[train['LotConfig']=='Inside','LotConfig']=0
	train.loc[train['LotConfig']=='Corner','LotConfig']=1
	train.loc[train['LotConfig']=='CulDSac','LotConfig']=2
	train.loc[train['LotConfig']=='FR2','LotConfig']=3
	train.loc[train['LotConfig']=='FR3','LotConfig']=4

	train.loc[train['LandSlope']=='Gtl','LandSlope']=0
	train.loc[train['LandSlope']=='Mod','LandSlope']=1
	train.loc[train['LandSlope']=='Sev','LandSlope']=2

	train.loc[train['Neighborhood']=='Blmngtn','Neighborhood']=0
	train.loc[train['Neighborhood']=='Blueste','Neighborhood']=1
	train.loc[train['Neighborhood']=='BrDale','Neighborhood']=2
	train.loc[train['Neighborhood']=='ClearCr','Neighborhood']=3
	train.loc[train['Neighborhood']=='CollgCr','Neighborhood']=4
	train.loc[train['Neighborhood']=='Crawfor','Neighborhood']=5
	train.loc[train['Neighborhood']=='Edwards','Neighborhood']=6
	train.loc[train['Neighborhood']=='Gilbert','Neighborhood']=7
	train.loc[train['Neighborhood']=='IDOTRR','Neighborhood']=8
	train.loc[train['Neighborhood']=='MeadowV','Neighborhood']=9
	train.loc[train['Neighborhood']=='Mitchel','Neighborhood']=10
	train.loc[train['Neighborhood']=='Names','Neighborhood']=11
	train.loc[train['Neighborhood']=='NAmes','Neighborhood']=11
	train.loc[train['Neighborhood']=='NoRidge','Neighborhood']=12
	train.loc[train['Neighborhood']=='NPkVill','Neighborhood']=13
	train.loc[train['Neighborhood']=='NridgHt','Neighborhood']=14
	train.loc[train['Neighborhood']=='NWAmes','Neighborhood']=15
	train.loc[train['Neighborhood']=='OldTown','Neighborhood']=16
	train.loc[train['Neighborhood']=='SWISU','Neighborhood']=17
	train.loc[train['Neighborhood']=='Sawyer','Neighborhood']=18
	train.loc[train['Neighborhood']=='SawyerW','Neighborhood']=19
	train.loc[train['Neighborhood']=='Somerst','Neighborhood']=20
	train.loc[train['Neighborhood']=='StoneBr','Neighborhood']=21
	train.loc[train['Neighborhood']=='Timber','Neighborhood']=22	
	train.loc[train['Neighborhood']=='Veenker','Neighborhood']=23
	train.loc[train['Neighborhood']=='BrkSide','Neighborhood']=23

	train.loc[train['Condition1']=='Artery','Condition1']=0
	train.loc[train['Condition1']=='Feedr','Condition1']=1
	train.loc[train['Condition1']=='Norm','Condition1']=2
	train.loc[train['Condition1']=='RRNn','Condition1']=3
	train.loc[train['Condition1']=='RRAn','Condition1']=4
	train.loc[train['Condition1']=='PosN','Condition1']=5
	train.loc[train['Condition1']=='PosA','Condition1']=6
	train.loc[train['Condition1']=='RRNe','Condition1']=7
	train.loc[train['Condition1']=='RRAe','Condition1']=8

	train.loc[train['Condition2']=='Artery','Condition2']=0
	train.loc[train['Condition2']=='Feedr','Condition2']=0
	train.loc[train['Condition2']=='Norm','Condition2']=0
	train.loc[train['Condition2']=='RRNn','Condition2']=0
	train.loc[train['Condition2']=='RRAn','Condition2']=0
	train.loc[train['Condition2']=='PosN','Condition2']=0
	train.loc[train['Condition2']=='PosA','Condition2']=0
	#train.loc[train['Condition2']=='RRNe','Condition2']=0
	#train.loc[train['Condition2']=='RRAe','Condition2']=0
	
	train.loc[train['BldgType']=='1Fam','BldgType']=0
	train.loc[train['BldgType']=='2FmCon','BldgType']=1
	train.loc[train['BldgType']=='2fmCon','BldgType']=1
	train.loc[train['BldgType']=='Duplex','BldgType']=2
	train.loc[train['BldgType']=='TwnhsE','BldgType']=3
	train.loc[train['BldgType']=='Twnhs','BldgType']=3
	#train.loc[train['BldgType']=='TwnhsI','BldgType']=4

	train.loc[train['HouseStyle']=='1Story','HouseStyle']=0
	train.loc[train['HouseStyle']=='1.5Fin','HouseStyle']=1
	train.loc[train['HouseStyle']=='1.5Unf','HouseStyle']=2
	train.loc[train['HouseStyle']=='2Story','HouseStyle']=3
	train.loc[train['HouseStyle']=='2.5Fin','HouseStyle']=4
	train.loc[train['HouseStyle']=='2.5Unf','HouseStyle']=5
	train.loc[train['HouseStyle']=='SFoyer','HouseStyle']=6		
	train.loc[train['HouseStyle']=='SLvl','HouseStyle']=7

	train.loc[train['RoofStyle']=='Flat','RoofStyle']=0
	train.loc[train['RoofStyle']=='Gable','RoofStyle']=1
	train.loc[train['RoofStyle']=='Gambrel','RoofStyle']=2
	train.loc[train['RoofStyle']=='Hip','RoofStyle']=3
	train.loc[train['RoofStyle']=='Mansard','RoofStyle']=4
	train.loc[train['RoofStyle']=='Shed','RoofStyle']=5

	train.loc[train['RoofMatl']=='ClyTile','RoofMatl']=0
	train.loc[train['RoofMatl']=='CompShg','RoofMatl']=1
	train.loc[train['RoofMatl']=='Membran','RoofMatl']=2
	train.loc[train['RoofMatl']=='Metal','RoofMatl']=3
	train.loc[train['RoofMatl']=='Roll','RoofMatl']=4
	train.loc[train['RoofMatl']=='Tar&Grv','RoofMatl']=5
	train.loc[train['RoofMatl']=='WdShake','RoofMatl']=6	
	train.loc[train['RoofMatl']=='WdShngl','RoofMatl']=7

	train.loc[train['Exterior1st']=='AsbShng','Exterior1st']=0
	train.loc[train['Exterior1st']=='AsphShn','Exterior1st']=1
	train.loc[train['Exterior1st']=='BrkComm','Exterior1st']=2
	train.loc[train['Exterior1st']=='Brk Cmn','Exterior1st']=2
	train.loc[train['Exterior1st']=='BrkFace','Exterior1st']=3
	train.loc[train['Exterior1st']=='CBlock','Exterior1st']=4
	train.loc[train['Exterior1st']=='HdBoard','Exterior1st']=5	
	train.loc[train['Exterior1st']=='ImStucc','Exterior1st']=6
	train.loc[train['Exterior1st']=='MetalSd','Exterior1st']=7
	train.loc[train['Exterior1st']=='Other','Exterior1st']=8
	train.loc[train['Exterior1st']=='Plywood','Exterior1st']=9
	train.loc[train['Exterior1st']=='PreCast','Exterior1st']=10
	train.loc[train['Exterior1st']=='Stone','Exterior1st']=11
	train.loc[train['Exterior1st']=='Stucco','Exterior1st']=12
	train.loc[train['Exterior1st']=='VinylSd','Exterior1st']=13
	train.loc[train['Exterior1st']=='Wd Sdng','Exterior1st']=14
	train.loc[train['Exterior1st']=='WdShing','Exterior1st']=15
	train.loc[train['Exterior1st']=='CemntBd','Exterior1st']=16
	#train.loc[train['Exterior1st']=='Wd Shng','Exterior1st']=17

	train.loc[train['Exterior2nd']=='AsbShng','Exterior2nd']=0
	train.loc[train['Exterior2nd']=='AsphShn','Exterior2nd']=1
	train.loc[train['Exterior2nd']=='BrkComm','Exterior2nd']=2
	train.loc[train['Exterior2nd']=='Brk Cmn','Exterior2nd']=2
	train.loc[train['Exterior2nd']=='BrkFace','Exterior2nd']=3
	train.loc[train['Exterior2nd']=='CBlock','Exterior2nd']=4
	train.loc[train['Exterior2nd']=='HdBoard','Exterior2nd']=5	
	train.loc[train['Exterior2nd']=='ImStucc','Exterior2nd']=6
	train.loc[train['Exterior2nd']=='MetalSd','Exterior2nd']=7
	train.loc[train['Exterior2nd']=='Other','Exterior2nd']=8
	train.loc[train['Exterior2nd']=='Plywood','Exterior2nd']=9
	train.loc[train['Exterior2nd']=='PreCast','Exterior2nd']=10
	train.loc[train['Exterior2nd']=='Stone','Exterior2nd']=11
	train.loc[train['Exterior2nd']=='Stucco','Exterior2nd']=12
	train.loc[train['Exterior2nd']=='VinylSd','Exterior2nd']=13
	train.loc[train['Exterior2nd']=='Wd Sdng','Exterior2nd']=14
	train.loc[train['Exterior2nd']=='WdShing','Exterior2nd']=15
	train.loc[train['Exterior2nd']=='CmentBd','Exterior2nd']=16
	train.loc[train['Exterior2nd']=='Wd Shng','Exterior2nd']=17

	train.loc[train['MasVnrType']=='BrkCmn','MasVnrType']=0
	train.loc[train['MasVnrType']=='BrkFace','MasVnrType']=1
	train.loc[train['MasVnrType']=='CBlock','MasVnrType']=2
	train.loc[train['MasVnrType']=='None','MasVnrType']=3
	train.loc[train['MasVnrType']=='Stone','MasVnrType']=4

	train.loc[train['ExterQual']=='Ex','ExterQual']=0
	train.loc[train['ExterQual']=='Gd','ExterQual']=1
	train.loc[train['ExterQual']=='TA','ExterQual']=2
	train.loc[train['ExterQual']=='Fa','ExterQual']=3
	#train.loc[train['ExterQual']=='Po','ExterQual']=4

	train.loc[train['ExterCond']=='Ex','ExterCond']=0
	train.loc[train['ExterCond']=='Gd','ExterCond']=1
	train.loc[train['ExterCond']=='TA','ExterCond']=2
	train.loc[train['ExterCond']=='Fa','ExterCond']=3
	train.loc[train['ExterCond']=='Po','ExterCond']=4

	train.loc[train['Foundation']=='BrkTil','Foundation']=0
	train.loc[train['Foundation']=='CBlock','Foundation']=1
	train.loc[train['Foundation']=='PConc','Foundation']=2
	train.loc[train['Foundation']=='Slab','Foundation']=3
	train.loc[train['Foundation']=='Stone','Foundation']=4
	train.loc[train['Foundation']=='Wood','Foundation']=5

	train.loc[train['BsmtQual']=='Ex','BsmtQual']=0
	train.loc[train['BsmtQual']=='Gd','BsmtQual']=1
	train.loc[train['BsmtQual']=='TA','BsmtQual']=2
	train.loc[train['BsmtQual']=='Fa','BsmtQual']=3
	#train.loc[train['BsmtQual']=='Po','BsmtQual']=4
	#train.loc[train['BsmtQual']=='NA','BsmtQual']=5
	
	train.loc[train['BsmtCond']=='Ex','BsmtCond']=0
	train.loc[train['BsmtCond']=='Gd','BsmtCond']=1
	train.loc[train['BsmtCond']=='TA','BsmtCond']=2
	train.loc[train['BsmtCond']=='Fa','BsmtCond']=3
	train.loc[train['BsmtCond']=='Po','BsmtCond']=4
	#train.loc[train['BsmtCond']=='NA','BsmtCond']=5

	train.loc[train['BsmtExposure']=='Gd','BsmtExposure']=0
	train.loc[train['BsmtExposure']=='Av','BsmtExposure']=1
	train.loc[train['BsmtExposure']=='Mn','BsmtExposure']=2
	train.loc[train['BsmtExposure']=='No','BsmtExposure']=3
	#train.loc[train['BsmtExposure']=='NA','BsmtExposure']=4

	train.loc[train['BsmtFinType1']=='GLQ','BsmtFinType1']=0
	train.loc[train['BsmtFinType1']=='ALQ','BsmtFinType1']=1
	train.loc[train['BsmtFinType1']=='BLQ','BsmtFinType1']=2
	train.loc[train['BsmtFinType1']=='LwQ','BsmtFinType1']=3
	train.loc[train['BsmtFinType1']=='Unf','BsmtFinType1']=4
	train.loc[train['BsmtFinType1']=='Rec','BsmtFinType1']=5
	#train.loc[train['BsmtFinType1']=='NA','BsmtFinType1']=5

	train.loc[train['BsmtFinType2']=='GLQ','BsmtFinType2']=0
	train.loc[train['BsmtFinType2']=='ALQ','BsmtFinType2']=1
	train.loc[train['BsmtFinType2']=='BLQ','BsmtFinType2']=2
	train.loc[train['BsmtFinType2']=='LwQ','BsmtFinType2']=3
	train.loc[train['BsmtFinType2']=='Unf','BsmtFinType2']=4
	train.loc[train['BsmtFinType2']=='Rec','BsmtFinType2']=5
	#train.loc[train['BsmtFinType2']=='NA','BsmtFinType2']=5

	train.loc[train['Heating']=='Floor','Heating']=0
	train.loc[train['Heating']=='GasA','Heating']=1
	train.loc[train['Heating']=='GasW','Heating']=2
	train.loc[train['Heating']=='Grav','Heating']=3
	train.loc[train['Heating']=='OthW','Heating']=4
	train.loc[train['Heating']=='Wall','Heating']=5

	train.loc[train['HeatingQC']=='Ex','HeatingQC']=0
	train.loc[train['HeatingQC']=='Gd','HeatingQC']=1
	train.loc[train['HeatingQC']=='TA','HeatingQC']=2
	train.loc[train['HeatingQC']=='Fa','HeatingQC']=3
	train.loc[train['HeatingQC']=='Po','HeatingQC']=4

	train.loc[train['CentralAir']=='N','CentralAir']=0
	train.loc[train['CentralAir']=='Y','CentralAir']=1

	train.loc[train['Electrical']=='SBrkr','Electrical']=0
	train.loc[train['Electrical']=='FuseA','Electrical']=1
	train.loc[train['Electrical']=='FuseF','Electrical']=2
	train.loc[train['Electrical']=='FuseP','Electrical']=3
	#train.loc[train['Electrical']=='Mix','Electrical']=4

	train.loc[train['KitchenQual']=='Ex','KitchenQual']=0
	train.loc[train['KitchenQual']=='Gd','KitchenQual']=1
	train.loc[train['KitchenQual']=='TA','KitchenQual']=2
	train.loc[train['KitchenQual']=='Fa','KitchenQual']=3
	#train.loc[train['KitchenQual']=='Po','KitchenQual']=4

	train.loc[train['Functional']=='Typ','Functional']=0
	train.loc[train['Functional']=='Min1','Functional']=1
	train.loc[train['Functional']=='Min2','Functional']=2
	train.loc[train['Functional']=='Mod','Functional']=3
	train.loc[train['Functional']=='Maj1','Functional']=4
	train.loc[train['Functional']=='Maj2','Functional']=5
	train.loc[train['Functional']=='Sev','Functional']=6
	#train.loc[train['Functional']=='Sal','Functional']=7

	train.loc[train['FireplaceQu']=='Ex','FireplaceQu']=0
	train.loc[train['FireplaceQu']=='Gd','FireplaceQu']=1
	train.loc[train['FireplaceQu']=='TA','FireplaceQu']=2
	train.loc[train['FireplaceQu']=='Fa','FireplaceQu']=3
	train.loc[train['FireplaceQu']=='Po','FireplaceQu']=4
	#train.loc[train['FireplaceQu']=='NA','FireplaceQu']=5

	train.loc[train['GarageType']=='2Types','GarageType']=0
	train.loc[train['GarageType']=='Attchd','GarageType']=1
	train.loc[train['GarageType']=='Basment','GarageType']=2
	train.loc[train['GarageType']=='BuiltIn','GarageType']=3
	train.loc[train['GarageType']=='CarPort','GarageType']=4
	train.loc[train['GarageType']=='Detchd','GarageType']=5
	#train.loc[train['GarageType']=='NA','GarageType']=6

	train.loc[train['GarageFinish']=='Fin','GarageFinish']=0
	train.loc[train['GarageFinish']=='RFn','GarageFinish']=1
	train.loc[train['GarageFinish']=='Unf','GarageFinish']=3
	#train.loc[train['GarageFinish']=='NA','GarageFinish']=4

	train.loc[train['GarageQual']=='Ex','GarageQual']=0
	train.loc[train['GarageQual']=='Gd','GarageQual']=1
	train.loc[train['GarageQual']=='TA','GarageQual']=2
	train.loc[train['GarageQual']=='Fa','GarageQual']=3
	train.loc[train['GarageQual']=='Po','GarageQual']=4
	#train.loc[train['GarageQual']=='NA','GarageQual']=5

	train.loc[train['GarageCond']=='Ex','GarageCond']=0
	train.loc[train['GarageCond']=='Gd','GarageCond']=1
	train.loc[train['GarageCond']=='TA','GarageCond']=2
	train.loc[train['GarageCond']=='Fa','GarageCond']=3
	train.loc[train['GarageCond']=='Po','GarageCond']=4
	#train.loc[train['GarageCond']=='NA','GarageCond']=5

	train.loc[train['PavedDrive']=='Y','PavedDrive']=0
	train.loc[train['PavedDrive']=='P','PavedDrive']=1
	train.loc[train['PavedDrive']=='N','PavedDrive']=2

	train.loc[train['PoolQC']=='Ex','PoolQC']=0
	train.loc[train['PoolQC']=='Gd','PoolQC']=1
	#train.loc[train['PoolQC']=='TA','PoolQC']=2
	#train.loc[train['PoolQC']=='Fa','PoolQC']=3
	#train.loc[train['PoolQC']=='NA','PoolQC']=4

	train.loc[train['Fence']=='GdPrv','Fence']=0
	train.loc[train['Fence']=='MnPrv','Fence']=1
	train.loc[train['Fence']=='GdWo','Fence']=2
	train.loc[train['Fence']=='MnWw','Fence']=3
	#train.loc[train['Fence']=='NA','Fence']=4

	train.loc[train['MiscFeature']=='Elev','MiscFeature']=0
	train.loc[train['MiscFeature']=='Gar2','MiscFeature']=1
	train.loc[train['MiscFeature']=='Othr','MiscFeature']=2
	train.loc[train['MiscFeature']=='Shed','MiscFeature']=3
	#train.loc[train['MiscFeature']=='TenC','MiscFeature']=4
	#train.loc[train['MiscFeature']=='NA','MiscFeature']=5

	train.loc[train['SaleType']=='WD','SaleType']=0
	train.loc[train['SaleType']=='CWD','SaleType']=1
	train.loc[train['SaleType']=='VWD','SaleType']=2
	train.loc[train['SaleType']=='New','SaleType']=3
	train.loc[train['SaleType']=='COD','SaleType']=4
	train.loc[train['SaleType']=='Con','SaleType']=5
	train.loc[train['SaleType']=='ConLw','SaleType']=6
	train.loc[train['SaleType']=='ConLI','SaleType']=7
	train.loc[train['SaleType']=='ConLD','SaleType']=8
	train.loc[train['SaleType']=='Oth','SaleType']=9

	train.loc[train['SaleCondition']=='Normal','SaleCondition']=9
	train.loc[train['SaleCondition']=='Abnorml','SaleCondition']=9
	train.loc[train['SaleCondition']=='AdjLand','SaleCondition']=9
	train.loc[train['SaleCondition']=='Alloca','SaleCondition']=9
	train.loc[train['SaleCondition']=='Family','SaleCondition']=9
	train.loc[train['SaleCondition']=='Partial','SaleCondition']=9

	return train

def data_preparer(path,train_perc):
	'''prepare data before training'''
	'''train set processing'''
	pca = PCA(n_components=40)
	ss = MinMaxScaler()
	
	train = pd.read_csv(path+"train.csv", index_col = "Id").sample(frac=1)
	label=train.pop('SalePrice') #target
	train=data_wash_train(train)

	train = ss.fit_transform(train,label)
	train = pca.fit_transform(train)
	train_x=np.array(train).tolist() 
	train_y=np.array(label).tolist()
	divide=int(len(train_x)*train_perc)
	TrainSet=[train_x[0:divide],train_y[0:divide]]
	if train_perc==1.0:
		divide=int(len(train_x)*0.7)
	ValSet=[train_x[divide:-1],train_y[divide:-1]]

	#print(len(TrainSet))

	'''test set processing'''
	test = pd.read_csv(path+"test.csv")
	pid=np.array(test.pop('Id')).tolist()
	test=data_wash_test(test)
	test = ss.transform(test)
	test = pca.transform(test)
	test_x=np.array(test).tolist() # train input in list
	TestSet=[test_x,pid]

	return TrainSet,ValSet,TestSet


if __name__ == '__main__':
	#data_preparer('../input/',0.5)
	main()