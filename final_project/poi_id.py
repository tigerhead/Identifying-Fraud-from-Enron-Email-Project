#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
   






#########################################################
if __name__ == '__main__':
    
    ### Task 1: Select what features you'll use.
    ### please run poi_features_selection.py to see 
    ### the feature selection process
    ### The first feature must be "poi".
    ##this is 
    features_list = ['poi', 'from_poi_ratio', 'shared_receipt_with_poi', 'to_poi_ratio'  ] # You will need to use more features
    ### Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    from poi_feature_selection import explore_dataset
    explore_dataset(data_dict)
    from poi_feature_selection import clean_dataset
    data_dict = clean_dataset(data_dict)
    
    ##added three new features
    from poi_feature_selection import add_features
    data_dict = add_features(data_dict)
    
    #from feature selection process, 3 features including added feature optimal 
    from poi_feature_selection import feature_selection
    selected_features_list = feature_selection(data_dict, 3)
    print "Selected Features: ", selected_features_list
    features_list = ['poi'] + selected_features_list
    

    ### Task 2: Remove outliers
    
    ### from outlier removal process, keys to remove are identified
    keys_to_remove = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    from poi_outlier_removal import remove_outliers
    data_dict = remove_outliers(data_dict, keys_to_remove)
    
    ### Task 3: Create new feature(s), 
    ### implemented in poi_feature_selection.add_features
    
    
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ### Task 4: Try a varity of classifiers
    ### Please run poi_classifier_selection.py.    
    ### http://scikit-learn.org/stable/modules/pipeline.html

    
    # Provided to give you a starting point. Try a varity of classifiers.

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### Please run poi_classifier_tuning.py, poi_classifier_tuning_loops.py.    
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    
    
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=20)    

    test_classifier(clf, my_dataset, features_list)    

    ### Dump your classifier, dataset, and features_list so 
    ### anyone can run/check your results.
    
    dump_classifier_and_data(clf, my_dataset, features_list)
    #print len(features_train[0])
