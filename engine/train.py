import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher

from sklearn.metrics import roc_auc_score

# Testing more github stuff


#now I am going to add stuff to the main branch and try and get into the other one



def run(fold, model):
    #read training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    #training data is all the folds except the fold that is given 
    #also reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    #validiation data for a given fold is one where the fold is equall to kfold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #drop the target column from the dataframe and convert it to a numpy array
    x_train = df_train.drop(['failure', 'id'], axis=1).values
    y_train = df_train.failure.values

    #same for validataion:
    x_valid = df_valid.drop(['failure', 'id'], axis=1).values
    y_valid = df_valid.failure.values

    #the model is imported:
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    #create predictions and print the evaluations
    preds = clf.predict(x_valid)

    print(f'SUM OF PREDICTIONS: {sum(preds)}')
    print(f'CORRECT AMOUNT OF PREDS: {sum(y_valid)}')

    accuracy = roc_auc_score(y_valid, preds)
    print(f'Fold = {fold}, Accuracy = {accuracy}')

    #save the model.






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fold',
        type=int,
    )
    parser.add_argument(
        '--model',
        type=str,
    )

    args = parser.parse_args()

    run(
        fold = args.fold,
        model = args.model,
    )


    #python train.py --fold 0 --model rf