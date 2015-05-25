#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import pprint



#explore dataset
def explore_dataset(data_dict):    
    #Dataset size, structure, features
    print "dataset size:",  len(data_dict)
    print data_dict[data_dict.keys()[0]]
    print "Number of features:", len(data_dict[data_dict.keys()[0]])
    print "Feature names: ", data_dict[data_dict.keys()[0]].keys()
    #check Nan values
    poiCounter = 0
    salaryCounter = 0
    emailAddressCounter = 0
    nanTotalPaymentCounter = 0
    nanTotalPaymentPOI = 0
    nanMailToPoiCounter=0
    nanTotalStockCounter = 0

    for name in data_dict.keys():        
        if data_dict[name]["poi"]:            
            poiCounter +=  1
        if str(data_dict[name]['salary']) != 'NaN' :         
            salaryCounter +=  1
        if  data_dict[name]['email_address'] != None and data_dict[name]['email_address'] != 'NaN':
            emailAddressCounter +=  1         
        if  data_dict[name]['total_payments'] == None or data_dict[name]['total_payments'] == 'NaN':
            nanTotalPaymentCounter += 1       
        if data_dict[name]["poi"]:
            nanTotalPaymentPOI += 1
        if str(data_dict[name]['total_stock_value'])   == 'NaN':
            nanTotalStockCounter += 1    
        if str(data_dict[name]['from_this_person_to_poi'])   == 'NaN':
            nanMailToPoiCounter += 1
           
            
    print "Number of nonTotalPaymentPOI: ",   nanTotalPaymentPOI , " percentage: " ,   float(nanTotalPaymentPOI)/poiCounter           
    print "Number of NaN total payment: ",   nanTotalPaymentCounter , " percentage: " ,   float(nanTotalPaymentCounter)/len(data_dict)   
    print  "Number of POI: ", poiCounter
    print  "Number of non-NaN Salary: ", salaryCounter
    print  "email address: ", emailAddressCounter
    print  "Number of NaN total_stock_value: ", nanTotalStockCounter
    print  "from_this_person_to_poi: ", nanMailToPoiCounter  
    
    return


#explore dataset
def clean_dataset(data_dict):    
    #List of features which is number type
    feature_in_num = ['salary', 'to_messages', 'deferral_payments', 
                      'exercised_stock_options', 'total_payments', 
                      'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
                      'restricted_stock_deferred', 'total_stock_value', 
                      'expenses', 'loan_advances', 'from_messages', 
                      'other', 'from_this_person_to_poi', 
                      'director_fees', 'deferred_income', 'long_term_incentive', 
                      'from_poi_to_this_person']

    
     
    #set NaN value to 0
    for name in data_dict.keys():        
        for key in data_dict[name].keys():
            if str(data_dict[name][key]) == 'NaN':
                data_dict[name][key] = 0       
           
            
    
    #Total payment is a critical financial data, try to calculate total payment if it is zero after replacing NaN with zero
    totalPaymentMatchCounter = 0
    totalPaymentNotNaNCounter = 0
    totalPaymentFixed = 0
    for name in data_dict.keys(): 
        if data_dict[name]['total_payments'] == 0:
            data_dict[name]['total_payments'] = data_dict[name]['salary'] + \
                                                data_dict[name]['bonus']  + \
                                                data_dict[name]['deferral_payments'] + \
                                                data_dict[name]['expenses'] + \
                                                data_dict[name]['other'] + \
                                                data_dict[name]['long_term_incentive'] + \
                                                data_dict[name]['loan_advances'] + \
                                                data_dict[name]['deferred_income']
            # NaN total payment calculated
            if  data_dict[name]['total_payments'] > 0:
                totalPaymentFixed += 1
                                                    
        else:
            totalPaymentNotNaNCounter += 1
            total =   data_dict[name]['salary'] + \
                      data_dict[name]['bonus']  + \
                      data_dict[name]['deferral_payments'] + \
                      data_dict[name]['expenses'] + \
                      data_dict[name]['other'] + \
                      data_dict[name]['long_term_incentive'] + \
                      data_dict[name]['loan_advances'] + \
                      data_dict[name]['deferred_income'] 
                      
            if total ==  data_dict[name]['total_payments']:
                  totalPaymentMatchCounter += 1
                  
    print "Number of Non-NaN total payment:  ",   totalPaymentNotNaNCounter, \
          "    Percentage matching calculation: ",    float(totalPaymentMatchCounter)/float(totalPaymentNotNaNCounter)                                              
                                                
    print "Number of total payment fixed: ",  totalPaymentFixed
    
    return data_dict

  
    


#Add new features dataset, the function should be called after data is cleaned and all NaN nubmer is set to 0
def add_features(data_dict):    
    #Add three features: 
    #salary_total_pay_ratio = salary/total_payments
    #from_pio_ratio = from_poi_to_this_person/from_messages
    #to_pio_ratio = from_this_person_to_poi/to_messages  
    #total_stock_exstock_ratio = exercised_stock_options/total_stock_value
     
    #set NaN value to 0
    for name in data_dict.keys():        
        if  data_dict[name]['salary']  != 0:
            data_dict[name]['salary_total_pay_ratio'] = \
              float(data_dict[name]['salary'] ) /float(data_dict[name]['total_payments'])
            
        else:
             data_dict[name]['salary_total_pay_ratio'] = 0
             
             
        if  data_dict[name]['total_stock_value']  != 0:
            data_dict[name]['total_exstock_stock_ratio'] = \
              float(data_dict[name]['exercised_stock_options'])/float(data_dict[name]['total_stock_value'])  
        else:             
             data_dict[name]['total_exstock_stock_ratio'] = 0  
                   
        if  data_dict[name]['from_messages']  != 0:
            data_dict[name]['from_poi_ratio'] = \
              float(data_dict[name]['from_poi_to_this_person'])/float(data_dict[name]['from_messages'])
        else:
            data_dict[name]['from_poi_ratio'] = 0    
        
        if  data_dict[name]['to_messages']  != 0:
            data_dict[name]['to_poi_ratio'] = \
              float(data_dict[name]['from_this_person_to_poi'])/float(data_dict[name]['to_messages'])
        else:
            data_dict[name]['to_poi_ratio'] = 0
                
    return data_dict

def get_feature_list(data_dict):
    feature_list = data_dict[data_dict.keys()[0]].keys()
    for i in range(0 ,len(feature_list)):
        if(feature_list[i] == 'poi'):
            for l in range(0, i):
              feature_list[i - l] =  feature_list[i- l -1]
            feature_list[0] = 'poi' 
            break    
    return feature_list

def feature_selection(data_dict, k):
    
    feature_list = get_feature_list(data_dict)   
               
    # print "Auto generated feature list: " ,  feature_list 
    ### featureFormat can't handle string type, and email_adress is unique to 
    ### which should not be a training feature
    
    if 'email_address' in feature_list:
         feature_list.remove('email_address'); 
    ### Added  from_pio_ratio and to_pio_ratio, from_poi_to_this_person and   
    ### from_this_person_to_poi,from_poi_to_this_person, from_messages,to_messages,   
    ### should be removed
    if 'from_poi_to_this_person' in feature_list:
        feature_list.remove('from_poi_to_this_person')
        
    if 'from_this_person_to_poi' in feature_list:
        feature_list.remove('from_this_person_to_poi') 
        
    if 'from_messages' in feature_list:
        feature_list.remove('from_messages')
        
    if 'to_messages' in feature_list:
        feature_list.remove('to_messages')   
        
    if 'salary' in feature_list:
        feature_list.remove('salary')
        
    if 'total_payments' in feature_list:
        feature_list.remove('total_payments')
        
    if 'total_stock_value' in feature_list:
        feature_list.remove('total_stock_value')
        
    if 'exercised_stock_options' in feature_list:
        feature_list.remove('exercised_stock_options')       
                   
                 
    data = featureFormat(data_dict, feature_list, sort_keys = False)
    labels, features = targetFeatureSplit(data)
    selector = SelectKBest(f_classif, k)
    selector.fit(features, labels) 
    KBest_feature_list = [feature_list[i + 1] for i in selector.get_support(indices=True) ]

    
    return KBest_feature_list

#########################################################
if __name__ == '__main__':
    
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi', 'total_pay_salary_ratio',  'from_pio_ratio', 'to_pio_ratio',  ] # You will need to use more features
    #features_list = ['poi', 'total_payments',  'salary'  ]
    ### Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    explore_dataset(data_dict)
    data_dict = clean_dataset(data_dict)
    data_dict = add_features(data_dict)
    
    nb_list, dt_list, svm_list, feature_lists = [], [], [], []
    for k in range(2, len(data_dict[data_dict.keys()[0]]) - 10  ):
        selected_feature_list = feature_selection(data_dict, k)
        print "Selected Features: ", selected_feature_list         
        features_list = ['poi'] + selected_feature_list
         
        feature_lists.append(np.array(features_list))         

        ### Extract features and labels from dataset for local testing
        data = featureFormat(data_dict, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
    

        ### Naive Bayes Classifier 
        clf = GaussianNB()     
        accuracy, precision,  recall, f1, f2 =test_classifier(clf, data_dict, features_list)
        nb_list.append(np.array([k, accuracy, precision,  recall, f1, f2 ]))
    
        ### Decision Tree Classifier 
        clf = tree.DecisionTreeClassifier(min_samples_split=5 ) 
        clf.fit( features, labels )
        counter = 0 
        for im in clf.feature_importances_:    
           counter +=1  
           print "importance for ", features_list[counter], ":  ", im  
     
        accuracy, precision,  recall, f1, f2 = test_classifier(clf, data_dict, features_list)
        dt_list.append(np.array([k, accuracy, precision,  recall, f1, f2 ]))
        ### SVM Classifier 
        clf = svm.SVC(kernel='rbf', C=100, gamma= 0.1, class_weight = 'auto')
        accuracy, precision, recall, f1, f2 =test_classifier(clf, data_dict, features_list)
        svm_list.append(np.array([k, accuracy, precision,  recall, f1, f2 ]))
    
    #create dataframe to list accuracy, precision,  recall, f1, f2 for different feature selections
    cloumns_list = ["feature Nums", "accuracy", "precision",  "recall", "f1", "f2"]
    
    feature_df = pd.DataFrame(np.array(feature_lists)) 
    nb_df = pd.DataFrame(np.array(nb_list), columns = cloumns_list) 
    dt_df = pd.DataFrame(np.array(dt_list), columns = cloumns_list) 
    svm_df = pd.DataFrame(np.array(svm_list), columns = cloumns_list)
    print "Feature lists : "
    pprint.pprint(feature_lists)
    print "Naive Bayes cross validation list : "
    pprint.pprint(nb_df)
    print "Decision Tree cross validation list: " 
    pprint.pprint(dt_df)
    print "SVM cross validation Lists: "
    pprint.pprint(svm_df) 
        
        

    