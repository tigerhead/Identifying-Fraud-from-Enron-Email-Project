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
from pandas import DataFrame, Series

import numpy as np
import pprint
from poi_feature_selection import get_feature_list
from ggplot import *


def covert_to_df(data_dict, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to dataframe of features for ggplot
        remove_NaN=True will convert "NaN" string to 0.0
        remove_all_zeroes=True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes=True will omit any data points for which
            any of the features you seek are 0.0
    """
    
    feature_list = get_feature_list(data_dict)
    ###  can't handle string type
    if 'email_address' in feature_list:
         feature_list.remove('email_address');                  
   
    dictionary = data_dict
    return_list = []
    index_list = []

    if sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        append = False
        
        for feature in feature_list:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            all_zeroes = True
            for item in tmp_list:
                if item != 0 and item != "NaN":
                    append = True

        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            any_zeroes = False
            if 0 in tmp_list or "NaN" in tmp_list:
                append = False
        if append:            
                return_list.append( np.array(tmp_list) )
                index_list.append(key)        
              
    df_name_index = DataFrame(np.array(return_list), index=index_list,  columns=feature_list)
    df_num_index = DataFrame(np.array(return_list),  columns=feature_list)
        
    return df_name_index, df_num_index



"""Try to find outlier by starting with max value in a feature.
   It can has K number of round, scatter plot will be drawn for each round
   to observe if more outliers can be visualized.
   It will return a list of names to be removed as outlier
   Not working well, not used in final project
"""
def find_outlier_by_max(df_name_index, feature, k, remvoe_poi=False):    
    # remove outlier by finding the largest value    
    df = df_name_index
    key_to_remove = []  
    for i in range(0, k):
        max_value = max(df[feature])
        print "Max ", feature, ": ",  max_value, 
        df_rm = df[df[feature] == max_value]        
        for index, row in df_rm.iterrows():         
            print "Name: ", index, '   poi: ',   row['poi']   
            if not remvoe_poi and float(row['poi']) == 1.0:                 
                print "This is POI, do not remove:" , index        
            else:                            
                key_to_remove.append(index)
                print "Outlier to remove: ",  index                
            df  = df_name_index.drop(index)              
               
        plot = scatter_lot(df, 'salary', 'bonus') 
        print plot 
    
    return key_to_remove


"""Try to find outlier by percentile
"""
def find_outlier_by_percentile(df_num_index, df_name_index, x_feature, y_feature, p, remvoe_poi=False ):    
    
    x_percentile_v = np.percentile(df_num_index[x_feature], p) 
    y_percentile_v = np.percentile(df_num_index[y_feature], p)    
    print p, " percentile ", x_feature, ": ", x_percentile_v  
    print p, " percentile ", y_feature, ": ", y_percentile_v       
    
    df_rm_x = df_name_index[df_name_index[x_feature] > x_percentile_v ]
    df_rm_y = df_name_index[df_name_index[y_feature] > y_percentile_v] 
    key_to_remove = []
    
    for index, row in df_rm_x.iterrows():         
            print "Name: ", index, '   poi: ',   row['poi']   
            if not remvoe_poi and float(row['poi']) == 1.0:                 
                print "This is POI, do not remove:" , index        
            else:                            
                key_to_remove.append(index)
                print "Outlier to remove: ",  index
                
    for index, row in df_rm_y.iterrows():         
            print "Name: ", index, '   poi: ',   row['poi']   
            if not remvoe_poi and float(row['poi']) == 1.0:                 
                print "This is POI, do not remove:" , index        
            else:
                if index not in  key_to_remove:                                            
                    key_to_remove.append(index)
                    print "Outlier to remove: ",  index                             
    
    plot = scatter_lot(df_num_index, x_feature, y_feature) + \
           xlim(0,  np.percentile(df_num_index[x_feature], p) ) + \
           ylim(0 , np.percentile(df_num_index[y_feature], p) )           
               
    print plot
    
    return key_to_remove  

 

def scatter_lot(df, x_feature, y_feature):
    
    plot = ggplot(aes(x=x_feature, y = y_feature , fill = 'poi'), data = df) + \
    geom_point()  + \
    ggtitle('' +  y_feature + ' vs '  + x_feature) 
    
    return plot

"""Remove outliers by their keys. The key list is identified by running this script program 
   and manually validation considering very small size of this dataset.   
"""

def remove_outliers(data_dict, key_to_remove):
    
    for key in key_to_remove:
        data_dict.pop(key)
        
    return data_dict


#########################################################
if __name__ == '__main__':
    
   
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    #create two dataframe, one indexed by name the key for data dictionary, one index by number for percentile calculation
    df_name_index, df_num_index = covert_to_df(data_dict)
    
    ###Check salary, bonus pair
    plot = scatter_lot(df_num_index,'salary', 'bonus')
    print plot   
    ktr_bonus_salry =  find_outlier_by_percentile \
    (df_num_index, df_name_index, 'salary', 'bonus', 98 )
    print "Potential outliers to remove by bonus and salary percentile", ktr_bonus_salry                     
    
    ###Check from_this_person_to_poi, from_poi_to_this_person pair
    plot = scatter_lot(df_num_index,'from_this_person_to_poi', 'from_poi_to_this_person')
    print plot 
    ktr_frompoi_topoi =  find_outlier_by_percentile \
    (df_num_index, df_name_index, 'from_this_person_to_poi', 'from_poi_to_this_person', 98 )    
    print "Potential outliers to remove by to_poi and from poi ", ktr_frompoi_topoi
    
    
    ###Check exercised_stock_options, total_stock_value pair
    plot = scatter_lot(df_num_index,  'exercised_stock_options', 'total_stock_value')
    print plot 
    ktr_exstock_totalstock =  find_outlier_by_percentile \
     (df_num_index, df_name_index, 'exercised_stock_options', 'total_stock_value', 98 ) 
    print "Potential outliers to remove by exercised_stock_options and total_stock_value", ktr_exstock_totalstock
    
    
     ###Check shared_receipt_with_poi, from_messages pair
    plot = scatter_lot(df_num_index,  'shared_receipt_with_poi', 'from_messages')
    print plot 
    ktr_sharepoi_frommsg =  find_outlier_by_percentile \
    (df_num_index, df_name_index, 'shared_receipt_with_poi', 'from_messages', 98 ) 
    print "Potential outliers to remove by shared_receipt_with_poi and from_messages", ktr_sharepoi_frommsg
    
   
   
    
    

    
    
        
        

    