import re
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import linear_model, model_selection, svm, ensemble, preprocessing, pipeline, tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, f1_score
from imblearn.combine import SMOTETomek 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
from skopt import BayesSearchCV

def load_data():
   
    # read data 
  try:
    data = pd.read_excel(r'C:\Users\Julien\OneDrive\ETH\BScThesis\03_Thesis_Valentina\automatic_detection.xls', header=None)
  except: 
    data = pd.read_excel(r'C:\Users\Mögu\OneDrive\ETH\BScThesis\03_Thesis_Valentina\automatic_detection.xls', header=None) 
 

  # read data 
  X = data.iloc[1:, 1:16]
  y = data.iloc[1:, 18]
  return(X,y)

def resample_smt(X,y):
  smt = SMOTETomek(random_state=42)
  X_res, y_res = smt.fit_resample(X, y)
  return(X_res, y_res)

def create_pipeline(classifier):
  # Define the pipeline
  steps=[]
  if classifier == 'BaggingClassifier':
    steps = [
        ('scaler', StandardScaler() if feature_scaling == 1 else None),       # Scale the features
        ('classifier', BaggingClassifier(estimator=SVC(), n_estimators=100))  # BaggingClassifier with SVM base estimator
    ]

  elif classifier == 'GradientBoostingClassifier':
    steps = [
    ('scaler', StandardScaler() if feature_scaling == 1 else None),  # Scale the features
    ('classifier', GradientBoostingClassifier(n_estimators=100))     # Boosting Classifier with SVM base estimator
    ]
        
  pipeline = Pipeline(steps)
  return pipeline

def perform_grid_search(X_train, y_train, pipeline, k, classifier):
    # Define hyperparameters to tune
  param_grid =  {}

  if classifier == 'BaggingClassifier':
    param_grid = {
        'classifier__estimator__C': [10],
        'classifier__estimator__kernel': [ 'rbf'],
        'classifier__n_estimators': [200]
    }
    
  elif classifier == 'GradientBoostingClassifier':
        param_grid = {
        'classifier__max_depth' : [10],
        'classifier__max_features' :['sqrt','log2'],
        'classifier__n_estimators': [100]
    }

  # Create GridSearchCV object
  grid_search =BayesSearchCV(pipeline, param_grid, cv=k, scoring=['accuracy', 'f1_macro', 'matthews_corrcoef'], n_jobs=-1, refit = 'accuracy', verbose = 2)

  # Fit the grid search to the data
  grid_search.fit(X_train, y_train)

  grid_search.cv_results_

  return grid_search


if __name__ == "__main__":
  
  X,y = load_data()

  label_enc = preprocessing.LabelEncoder()
  y = label_enc.fit_transform(y)

  x_train,y_train = resample_smt(X,y)
  
  list_of_ks = [5,10,20]
  list_of_classifiers = ["BaggingClassifier", "GradientBoostingClassifier"]
  # Perform grid search and get the best classifier
  for k in list_of_ks:
    for classifier in list_of_classifiers:
      pipe = create_pipeline(classifier)
      best_classifier = perform_grid_search(x_train, y_train, pipe, k, classifier)

       file_name=f'{classifier}_k{k}_exp'
      

      with open(file_name, 'wb') as file:
        pickle.dump(best_classifier, file)

      df = pd.DataFrame(best_classifier.cv_results_)

      # Specify the Excel file path
      excel_file_path = f'cv_results_k{k}_{classifier}_exp.xlsx'

      with pd.ExcelWriter(excel_file_path) as writer:
        df.to_excel(writer, index = 'False')

    std_acc = best_classifier.cv_results_['std_test_accuracy']
    mean_acc = best_classifier.cv_results_['mean_test_accuracy']
    mean_f1 = best_classifier.cv_results_['mean_test_f1_macro']
    std_f1 = best_classifier.cv_results_['std_test_f1_macro']
    mean_mcc = best_classifier.cv_results_['mean_test_matthews_corrcoef']
    std_mcc = best_classifier.cv_results_['std_test_matthews_corrcoef']

    print(f'k: {k}, Classifier: {classifier}, Accuracy:{mean_acc}, std of accuracy{std_acc}, f1: {mean_f1}, std of f1: {std_f1}, MCC: {mean_mcc}, std of mcc: {std_mcc}, Best prameters: {best_classifier.best_params_}')



  


   

