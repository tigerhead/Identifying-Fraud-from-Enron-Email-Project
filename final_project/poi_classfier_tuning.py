#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from time import time

def tune_classfier(clf, labels, features, parameters):
    
    tuner = GridSearchCV(clf, parameters, scoring='f1')
    ##tuner = GridSearchCV(clf, parameters, scoring='precision')
    ##tuner = GridSearchCV(clf, parameters, scoring='recall')
    ## tuner = GridSearchCV(clf, parameters)
    tuner.fit(features, labels )
    
    best_estimator= tuner.best_estimator_
    return best_estimator






#########################################################
if __name__ == '__main__':
    
    
    
    ###Tuning SVM
    selected_features_list = ['from_poi_ratio', 'salary_total_pay_ratio', 'shared_receipt_with_poi', 'loan_advances', 'to_poi_ratio']
                    
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    from poi_feature_selection import explore_dataset
    explore_dataset(data_dict)
    from poi_feature_selection import clean_dataset
    data_dict = clean_dataset(data_dict)
    
    ##added three new features
    from poi_feature_selection import add_features
    data_dict = add_features(data_dict)    
  
    print "Selected Features: ", selected_features_list
    features_list = ['poi'] + selected_features_list   
    
    ### Remove outlier, keys to remove are identified
    keys_to_remove = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    from poi_outlier_removal import remove_outliers
    data_dict = remove_outliers(data_dict, keys_to_remove)     

    ### Extract features and labels from dataset for local testing
    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    scaler = MinMaxScaler()
    #features = scaler.fit_transform(features )
    pca = PCA(copy=True, n_components=None,  whiten=False)
    svm = SVC(C=10, cache_size=200, class_weight='auto',
    coef0=0.0, degree=3, gamma=0.0, kernel='rbf'  )
    ## pipeline with scaler, PCA and SVM classifier
    clf = Pipeline(steps=[ ('scale', scaler),                    
                           ('reduce_dim', pca),  
                           ('svm', svm)])
    
    
    parameters =  dict(reduce_dim__n_components = [2, 3, 4],
                       svm__C = [1, 5, 10, 20, 50, 100, 500, 1000, 1e4, 1e5, 1e6],
                       svm__gamma = [ 0.01, 0.1, 1, 5], 
                       svm__class_weight = [{1:2}, {1:5}, {1:8}, {1:10}, {1:20}] )    
    
    clf = tune_classfier(clf, labels, features, parameters)    
    print 'Best SVM estimator selected by grid search: ', clf 
    
    t0 = time()    
    test_classifier(clf, data_dict, features_list)
    print "training  and validation time: ", round(time() - t0, 3), "s"
   
    
    ###Tuning Decision Tree
     ### Extract features and labels from dataset for local testing
    selected_features_list = ['from_poi_ratio', 'shared_receipt_with_poi', 'to_poi_ratio']    
    print "Selected Features: ", selected_features_list
    features_list = ['poi'] + selected_features_list 
    
    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data) 
    clf = DecisionTreeClassifier(compute_importances=True, random_state = 37)
    parameters =  { 'min_samples_split': [1, 2, 3, 4, 5, 10, 20, 50, 100]}
    clf = tune_classfier(clf, labels, features, parameters)   
    print 'Best DT estimator selected by grid search: ', clf
    t0 = time()     
    test_classifier(clf, data_dict, features_list)
    print "training  and validation time: ", round(time() - t0, 3), "s"
   
