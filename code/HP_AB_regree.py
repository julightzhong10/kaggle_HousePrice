import numpy as np 
import pandas as pd
from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import math

def main():

	'''Management Data'''
	data_path='../input/'
	model_folder='../model/AB/'
	'''Management Data'''
	'''Data prepare, rotate or shuffle'''
	TrainSet,ValSet,TestSet=data_preparer(data_path,1.0)
	clf = ensemble.AdaBoostRegressor(loss='square',n_estimators=600,learning_rate=1.0)
	#n_estimators=200,min_samples_leaf=7,max_depth=5,max_features='auto',oob_score=True :
	clf.fit(TrainSet[0],TrainSet[1])
	val_x=clf.predict(ValSet[0])
	print(math.sqrt(validation_accu(val_x,ValSet[1])))
	data_Id=TestSet[1]
	# print(TestSet[0])
	data_label=clf.predict(TestSet[0])
	data_result={'Id':data_Id,'SalePrice':data_label}
	df = pd.DataFrame(data_result, columns= ['Id', 'SalePrice'])
	export_csv = df.to_csv (model_folder+'result.csv', index = None, header=True)


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
	
	train = pd.read_csv(path+"train.csv", index_col = "Id")
	label=train.pop('SalePrice') #target
	train=data_wash_train(train)

	#train = ss.fit_transform(train,label)
	#train = pca.fit_transform(train)
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
	#test = ss.transform(test)
	#test = pca.transform(test)
	test_x=np.array(test).tolist() # train input in list
	TestSet=[test_x,pid]

	return TrainSet,ValSet,TestSet


if __name__ == '__main__':
	#data_preparer('../input/',0.5)
	main()