# Identifying-Fraud-from-Enron-Email-Project
4th project of Udacity Data Analyst Nanodegree

Final Result:
New Features Added:
salary_total_pay_ratio = salary/total_payments
from_pio_ratio = from_poi_to_this_person/from_messages
to_pio_ratio = from_this_person_to_poi/to_messages  
total_stock_exstock_ratio = exercised_stock_options/total_stock_value

Feature selected: 'from_poi_ratio', 'shared_receipt_with_poi', 'to_poi_ratio'
DecisionTreeClassifier(compute_importances=None, criterion='gini',       max_depth=None, max_features=None, max_leaf_nodes=None,   min_density=None, min_samples_leaf=1, min_samples_split=20, random_state=37, splitter='best')
Accuracy: 0.84856	Precision: 0.65145	Recall: 0.68500	F1: 0.66780	F2: 0.67802  Total predictions: 9000	True positives: 1370	False positives:  733	False negatives:  630 True negatives: 6267

Python files includes:
poi_feature_selection            feature slection process 
poi_classfier_selection.py       algorithm slection process 
poi_outlier_removal.py           outlier detection and removal
poi_classfier_tuning.py          classifer parameters tuning
poi_classfier_tuning_loops       classifer parameters finer tuning
poi_id.py                        main program to show case
