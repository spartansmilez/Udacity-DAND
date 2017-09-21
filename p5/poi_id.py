#!/usr/bin/python

import sys
import pickle
import random
import matplotlib.pyplot as plt
from time import time
import numpy as np
from numpy import mean
sys.path.append("../tools/")

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import *
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

print 'Number of features: ', len(features_list)
### Load the dictionary containing the dataset.
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Exploration of the dataset.
total_number_people = len(data_dict)
print 'Number of people in the dataset:', total_number_people

### Number of POI in the dataset.
num_poi = 0
for i in data_dict:
    if data_dict[i]['poi']==True:
        num_poi=num_poi+1
print 'Number of poi in the dataset:', num_poi
print 'Number of person who is not poi:', len(data_dict)-num_poi

### Task 2: Remove outliers.
features =['salary', 'bonus']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')

plt.show() # Plot showing that there is an outlier.

for i, j in data_dict.items():
    if j['salary'] != 'NaN' and j['salary'] > 10000000:
        print i
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0 )
data_dict.pop('LOCKHART EUGENE E', 0)
print "Number of datapoints after removing outliers is :", len(data_dict)

features =['salary', 'bonus']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')

plt.show()

### Task 3: Create new feature(s).
### Store to my_dataset for easy export below.
my_dataset = data_dict

for key, value in my_dataset.iteritems():
    value['from_poi_to_this_person_ratio']=0
    if value['to_messages'] and value['from_poi_to_this_person']!='NaN':
        value['from_poi_to_this_person']=float(value['from_poi_to_this_person'])
        value['to_messaages']=float(value['to_messages'])
        if value['from_poi_to_this_person'] > 0:
            value['from_poi_to_this_person_ratio']=value['from_poi_to_this_person']/value['to_messages']

for key, value in my_dataset.iteritems():
    value['from_this_person_to_poi_ratio']=0
    if value['from_messages'] and value['from_this_person_to_poi']!='NaN':
        value['from_this_person_to_poi']=float(value['from_this_person_to_poi'])
        value['from_messaages']=float(value['from_messages'])
        if value['from_this_person_to_poi'] > 0:
            value['from_this_person_to_poi_ratio']=value['from_this_person_to_poi']/value['from_messages']
                    
features_list.extend(['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio'])

print "I created two features: 'from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio'"

### Missing values in each feature.
nan = [0 for i in range(len(features_list))]
for i, person in my_dataset.iteritems():
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            nan[j] += 1
for i, feature in enumerate(features_list):
    print feature, nan[i]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale features via min-max
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
# featuer selection for GaussianNB
selection=SelectKBest(k=6)
selection.fit(features, labels)
# GaussianNB feature selection
#kbest_gnb = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#precision_gnb =[0.46, 0.47, 0.49, 0.5, 0.5, 0.52, 0.49, 0.49, 0.38, 0.37, 0.33, 0.32,
#              0.33, 0.33, 0.33, 0.25, 0.24, 0.22, 0.23, 0.24]
#recall_gnb = [0.32, 0.27, 0.35, 0.32, 0.33, 0.39, 0.38, 0.40, 0.32, 0.31, 0.31, 0.31,
#            0.31, 0.31, 0.31, 0.34, 0.33, 0.28, 0.29, 0.50]

#plt.plot(kbest_gnb, precision_gnb, 'b', label='Precision')
#plt.plot(kbest_gnb, recall_gnb, 'r', label='Recall')
#legend = plt.legend(loc='best', bbox_to_anchor=(1.45, 0.8), shadow=True)
#plt.xlabel('# of Features in SelectKBest')
#plt.ylabel('Cross-Validated Scores')
#plt.title('GaussianNB')
#plt.show()

# Random Forest feature selection
#kbest_rf=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#precision_rf = [0.27, 0.31, 0.60, 0.50, 0.45, 0.49, 0.46, 0.41, 0.39, 0.44, 0.43, 0.43,
#               0.48, 0.46, 0.46, 0.45, 0.44, 0.42, 0.41, 0.40]
#recall_rf = [0.28, 0.17, 0.31, 0.23, 0.22, 0.20, 0.17, 0.14, 0.14, 0.17, 0.15, 0.14,
#            0.16, 0.14, 0.14, 0.14, 0.15, 0.13, 0.12, 0.12]
#plt.plot(kbest_rf, precision_rf, 'b', label='Precision')
#plt.plot(kbest_rf, recall_rf, 'r', label='Recall')
#legend = plt.legend(loc='best', bbox_to_anchor=(1.45, 0.8), shadow=True)
#plt.xlabel('# of Features in SelectKBest')
#plt.ylabel('Cross-Validated Scores')
#plt.title('Random Forest')
#plt.show()

#K Nearest Neighbors feature selection
#kbest_knn=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#precision_knn = [0.68, 0.69, 0.64, 0.66, 0.64, 0.69, 0.61, 0.69, 0.64, 0.64, 0.64, 0.64,
#               0.64, 0.64, 0.64, 0.64, 0.64, 0.63, 0.63, 0.64]
#recall_knn = [0.19, 0.19, 0.26, 0.27, 0.28, 0.26, 0.21, 0.26, 0.17, 0.17, 0.17, 0.17,
#            0.17, 0.17, 0.17, 0.17, 0.17, 0.2, 0.2, 0.21]
#plt.plot(kbest_knn, precision_knn, 'b', label='Precision')
#plt.plot(kbest_knn, recall_knn, 'r', label='Recall')
#legend = plt.legend(loc='best', bbox_to_anchor=(1.45, 0.8), shadow=True)
#plt.xlabel('# of Features in SelectKBest')
#plt.ylabel('Cross-Validated Scores')
#plt.title('K Nearest Neighbors')
#plt.show()

#results = zip(selection.get_support(), features_list[1:], selection.scores_)
#results = sorted(results, key=lambda x: x[2], reverse=True)
#print "K-best features:", results

# update features list chosen manually and by SelectKBest
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
                 'salary', 'from_this_person_to_poi_ratio', 'deferred_income']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Tuning of classifiers to confirm best k.
# Potential pipeline steps.
scaler = MinMaxScaler()
select = SelectKBest()
gnb = GaussianNB()
rf = RandomForestClassifier()
knc = KNeighborsClassifier()

# Load pipeline steps into list.
steps = [
         # Preprocessing
         # ('min_max_scaler', scaler),
         
         # Feature selection
         ('feature_selection', select),
         
         # Classifier
         ('gnb', gnb)
         #('rf', rf)
         # ('knc', knc)
         ]

# Create pipeline.
pipeline = Pipeline(steps)

# Parameters to try in grid search.
parameters = dict(
                   feature_selection__k=[2, 3, 5, 6], 
                  # rf__criterion=['gini', 'entropy'],
                  # rf__max_depth=[None, 1, 2, 3, 4],
                  # rf__min_samples_split=[1, 2, 3, 4, 25],
                  # rf__min_samples_leaf=[1, 2, 3, 4],
                  # rf__min_weight_fraction_leaf=[0, 0.25, 0.5],
                  # rf__class_weight=[None, 'balanced'],
                  # rf__random_state=[42]
                  # knc__n_neighbors=[1, 2, 3, 4, 5],
                  # knc__leaf_size=[1, 10, 30, 60],
                  # knc__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
                  )
# Create training sets and test sets.
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Cross-validation for parameter tuning in grid search. 
sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )

# Create, fit, and make predictions with grid search.
gs = GridSearchCV(pipeline,
                  param_grid=parameters,
                  scoring="f1",
                  cv=sss,
                  error_score=0)
gs.fit(features_train, labels_train)
labels_predictions = gs.predict(features_test)

# Pick the classifier with the best tuned parameters.
clf = gs.best_estimator_
print "\n", "Best parameters are: ", gs.best_params_, "\n"

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Gaussian Naive Bayes
clf = GaussianNB()

# KNeighbors
#clf = KNeighborsClassifier()

# Random Forest
#clf = RandomForestClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
#tuned_parameters_knn = {"n_neighbors":[2, 5], "p":[2,3]}
#clf=GridSearchCV(clf, tuned_parameters_knn)

#tuned_parameters_ran = {"n_estimators":[2, 3, 5],  "criterion": ('gini', 'entropy')}
#clf = GridSearchCV(clf, tuned_parameters_ran)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print "accuracy: ", accuracy
print 'precision = ', precision_score(pred, labels_test)
print 'recall = ', recall_score(pred, labels_test)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)