import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import linear_model, model_selection, svm, ensemble, preprocessing, pipeline, tree
import numpy as np
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, f1_score
from imblearn.combine import SMOTETomek 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

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
  
def create_pipeline():

  # Define the pipeline
    steps = [
        ('scaler', StandardScaler() if feature_scaling == 1 else None ),  # Scale the features
        ('classifier', RandomForestClassifier())                          # BaggingClassifier with SVM base estimator
    ]
        
    pipeline = Pipeline(steps)
    return pipeline

def perform_grid_search(X_train, y_train, pipeline, k):

    # Define hyperparameters to tune
    param_grid = {
      'classifier__max_depth' : [1,5,10, None],
      'classifier__min_samples_split' : [2,4,6,10,100],
      'classifier__max_leaf_nodes' : [100,1000,None],
      'classifier__min_samples_leaf' : [1,10],
      'classifier__n_estimators' : [50,100,200,400],
      'classifier__max_features' : [1,4,8,15],
      'classifier__max_samples' : [0.1,0.5,1]
    }
   

    

    # Create GridSearchCV object
    grid_search = BayesSearchCV(pipeline, param_grid, cv=k, scoring=['accuracy', 'f1_macro', 'matthews_corrcoef'], n_jobs=-1, refit = 'accuracy', verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    grid_search.cv_results_

    return grid_search


if __name__ == "__main__":

  X,y = load_data()

  label_enc = preprocessing.LabelEncoder()
  y = label_enc.fit_transform(y)

  x_train, x_test,y_train,y_test = train_test_split(X,y, test_size=0.1, shuffle = True, random_state=42 )
  x_train,y_train = resample_smt(X,y)

  list_of_ks = [5,10,20]
   
  # Perform grid search and get the best classifier
  for k in list_of_ks:
    pipe = create_pipeline()
    best_classifier = perform_grid_search(x_train, y_train, pipe, k)


    file_name=f'xgb_k{k}_exp'
    

    with open(file_name, 'wb') as file:
      pickle.dump(best_classifier, file)

    df = pd.DataFrame(best_classifier.cv_results_)

    # Specify the Excel file path
    excel_file_path = f'cv_results_k{k}_rf_exp.xlsx'

    with pd.ExcelWriter(excel_file_path) as writer:
      df.to_excel(writer, index = 'False')


    std_acc = best_classifier.cv_results_['std_test_accuracy']
    mean_acc = best_classifier.cv_results_['mean_test_accuracy']
    mean_f1 = best_classifier.cv_results_['mean_test_f1_macro']
    std_f1 = best_classifier.cv_results_['std_test_f1_macro']
    mean_mcc = best_classifier.cv_results_['mean_test_matthews_corrcoef']
    std_mcc = best_classifier.cv_results_['std_test_matthews_corrcoef']

    print(f'k: {k}, Classifier: xgb,  Accuracy:{mean_acc}, std of accuracy{std_acc}, f1: {mean_f1}, std of f1: {std_f1}, MCC: {mean_mcc}, std of mcc: {std_mcc}, Best prameters: {best_classifier.best_params_}')



  


   

