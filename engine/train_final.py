from sklearn import ensemble
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

import config

#I am going to stop letting the passenger ID being included as it seems to be realied 
#but i do not understan why.

def run_final():
    df = pd.read_csv(config.TRAINING_FILE_TO_BE_FOLDED)

    x_train = df.drop(['failure', 'id'], axis=1).values
    y_train = df.failure.values

    x_test = pd.read_csv(config.TEST_FILE)
    x_test = x_test.drop(['id'], axis=1)

    #clf = SVC(class_weight=None, gamma='auto', C=1)
    #clf = ensemble.RandomForestClassifier(criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 200)
    clf = XGBClassifier()
    clf = ensemble.RandomForestClassifier()

    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)

    #converting into True and False
    #preds_tf = [True if i == 1 else False for i in preds] #this is not nessisary for the kaggle comp at the end 
   
    
    df_for_csv = pd.read_csv(config.TEST_FILE)
    prediction_csv = pd.DataFrame(df_for_csv['id'])
    prediction_csv['failure'] = preds #preds_tf
    prediction_csv.to_csv('/Users/ethan/Desktop/Ethan/Python/ML/framework/output/predictions_kaggle.csv', index=False)       


if __name__ == '__main__':
    run_final()


#This script got me an accuracy of 0.79425


#For the kaggle comp I am currently getting a score of 0.5 as this is not predicting any times wherer the data will fail