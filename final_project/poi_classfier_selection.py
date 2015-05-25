#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


#########################################################
if __name__ == '__main__':
    
   
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    from poi_feature_selection import explore_dataset
    explore_dataset(data_dict)
    from poi_feature_selection import clean_dataset
    data_dict = clean_dataset(data_dict)
    
    ##added 4 new features
    from poi_feature_selection import add_features
    data_dict = add_features(data_dict)
    
    ### from outlier removal process, keys to remove are identified
    keys_to_remove = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    from poi_outlier_removal import remove_outliers
    remove_outliers(data_dict, keys_to_remove)   
    
    #from feature selection process, 6 features including added feature optimal 
    for selected_features_list in [['from_poi_ratio', 'shared_receipt_with_poi', 'to_poi_ratio'], 
                                   ['from_poi_ratio', 'salary_total_pay_ratio',
                                     'shared_receipt_with_poi', 'to_poi_ratio'],
                                   ['from_poi_ratio', 'salary_total_pay_ratio',
                                    'shared_receipt_with_poi', 'loan_advances', 'to_poi_ratio']
                                   ]:
        
        print "Selected training features:  ", selected_features_list
    
        
        features_list = ['poi'] + selected_features_list    
    
        
        
  

        ### Extract features and labels from dataset for local testing
        data = featureFormat(data_dict, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
    
        ##Scale the features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features )    
    
        ##SVM
        from sklearn.svm import SVC
        svm = SVC(C=100, cache_size=200, class_weight='auto',
        coef0=0.0, degree=3, gamma=0.1, kernel='rbf'  )       
        test_classifier(svm, data_dict, features_list)
    
        ##Decision Tree     
        from sklearn import tree
        dt = tree.DecisionTreeClassifier(min_samples_split=10)     
        test_classifier(dt, data_dict, features_list)
    
        ##K Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier    
        knn = KNeighborsClassifier(n_neighbors=3, weights = 'distance' )
        test_classifier(knn, data_dict, features_list)
    
    
    
        ##Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10,  random_state=43)
        test_classifier(rf, data_dict, features_list)
    
        ## Ada Boost
        from sklearn.ensemble import AdaBoostClassifier
        adb = AdaBoostClassifier(n_estimators=10, random_state = 37)
        test_classifier(adb, data_dict, features_list)
    
    
    
   
    
   