
# Challenge Kaggle Forest cover
#MS BGD Novembre 2016
# Auteur :  Mahzad KALANTARI

import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

#Function to join soil types into 1 containing the existing soil type number
def get_soil_type(soil_types):
  for i in range(15, 55):
    if(soil_types[i]==1):
	    return i-14
#Function to join wilderness areas into 1
def get_wilderness_area(wilderness_areas):
	for i in range(11, 15):
		if(wilderness_areas[i]==1):
		    return i-10


train2 = pd.read_csv('train.csv')
test2 = pd.read_csv('test.csv')


colToScale=['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

train=train2.copy()
test=test2.copy()

for col in colToScale:
    scaler = preprocessing.StandardScaler().fit(train2[col].values.astype('float_'))
    train[col] = scaler.transform(train2[col].values.astype('float_'))
    test[col] = scaler.transform(test2[col].values.astype('float_'))




#Join soil types for training df
train['Soil_Type']=train.apply(get_soil_type, axis=1)

  #Join wilderness areas for training df
train['Wilderness_Area']=train.apply(get_wilderness_area, axis=1)

  #Join soil types for testing df
test['Soil_Type']=test.apply(get_soil_type, axis=1)
  #Join wilderness areas for testing df
test['Wilderness_Area']= test.apply(get_wilderness_area, axis=1)

test.fillna(0, None,0, True)
train.fillna(0, None,0, True)



def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180


train['Aspect2'] = train.Aspect.map(r)
test['Aspect2'] = test.Aspect.map(r)

train['Highwater'] = train.Vertical_Distance_To_Hydrology < 0
test['Highwater'] = test.Vertical_Distance_To_Hydrology < 0

train['EVDtH'] = train.Elevation-train.Vertical_Distance_To_Hydrology
test['EVDtH'] = test.Elevation-test.Vertical_Distance_To_Hydrology

train['EHDtH'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2
test['EHDtH'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2

train['Distanse_to_Hydrolody'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
test['Distanse_to_Hydrolody'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5

train['Hydro_Fire_1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
test['Hydro_Fire_1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']

train['Hydro_Fire_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
test['Hydro_Fire_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

train['Hydro_Road_1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

train['Hydro_Road_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])


train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology +
train.Horizontal_Distance_To_Roadways) / 3

test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology +
test.Horizontal_Distance_To_Roadways) / 3

train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2



feature_cols = [col for col in train.columns if col not in ['Cover_Type','Id']]

X_train = train[feature_cols]
X_test = test[feature_cols]
y = train['Cover_Type']
test_ids = test['Id']



dt = ensemble.ExtraTreesClassifier(n_estimators= 100, criterion='gini', max_depth=None,
    min_samples_split=2, min_samples_leaf=1, max_features='auto',
    bootstrap=False, oob_score=False, n_jobs=-1, random_state= None, verbose=0, class_weight='auto')


forest = ensemble.AdaBoostClassifier(dt, n_estimators=100, learning_rate=1)

forest.fit(X_train, y)

ypred_rf_opt_fe = forest.predict(X_test)

temp=test.copy()
temp['Cover_Type']= ypred_rf_opt_fe
temp=temp['Cover_Type']

'''
class_weights=pd.DataFrame({'Class_Count':temp.groupby(temp).agg(len)}, index=None)
print (class_weights)

class_weights['Class_Weights'] = temp.groupby(temp).agg(len)/len(temp)
sample_weights=class_weights.ix[y]

dt = ensemble.ExtraTreesClassifier(n_estimators= 300, criterion='gini', max_depth=None,
    min_samples_split=2, min_samples_leaf=1, max_features='auto',
    bootstrap=False, oob_score=False, n_jobs=-1, random_state=None,
     verbose=0, class_weight='auto')


forest = ensemble.AdaBoostClassifier(dt, n_estimators=40, learning_rate=1)


forest.fit(X_train, y)

y_pred= forest.predict(X_test)

'''

pd.DataFrame({'Id':test_ids,'Cover_Type':forest.predict(X_test)},
            columns=['Id','Cover_Type']).to_csv('MK44.csv',index=False)


print(pd.DataFrame(forest.feature_importances_,index=X_train.columns).sort([0], ascending=False) [:10])
