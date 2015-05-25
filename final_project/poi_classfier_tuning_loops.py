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
    
    scaler = MinMaxScaler()
    pca = PCA(copy=True, n_components=2,  whiten=False)
   
    
    for svm_c in [ 100, 500, 1000, 1e4]:
        for svm_gamma in [  0.1, 1, 5]: 
            for svm_class_weight in [{1:2}, {1:5}, {1:8}, {1:10}]:
                
               
                svm = SVC(C=svm_c, cache_size=200, class_weight=svm_class_weight,
                   coef0=0.0, degree=3, gamma=svm_gamma, kernel='rbf'  )
                 ## pipeline with scaler, PCA and SVM classifier
                clf = Pipeline(steps=[ ('scale', scaler),                    
                           ('reduce_dim', pca),  
                           ('svm', svm)])
                 
                print "C: ", svm_c, "  gamma: ", svm_gamma, "  class_weight: ", svm_class_weight
                t0 = time()
                test_classifier(clf, data_dict, features_list)
                print "training  and validation time: ", round(time() - t0, 3), "s"
   
        
   
    
   
    
    ###Tuning Decision Tree
     ### Extract features and labels from dataset for local testing
    selected_features_list = ['from_poi_ratio', 'shared_receipt_with_poi', 'to_poi_ratio']    
    print "Selected Features: ", selected_features_list
    features_list = ['poi'] + selected_features_list     
    
    for dt_min_samples_split in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        clf = DecisionTreeClassifier(min_samples_split = dt_min_samples_split , 
                                     random_state = 37)
        print "min_samples_split: ",  dt_min_samples_split
        t0 = time()     
        test_classifier(clf, data_dict, features_list)
        print "training  and validation time: ", round(time() - t0, 3), "s"
       
        
    
    
      
   
   
   
