# -*- coding: utf-8 -*-

#Created on Tue Oct 24 09:44:41 2017

#@author: yuxue 17319446

#for unzipping datasets and storing them in a temp dir
import config
import zipfile
import tempfile
#for importing external datasets
import numpy as np
import pandas as pd
#for visualizing data
import seaborn
import matplotlib.pyplot as plt
#prepossessing
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
#for using machine learning algorithms and doing ten cross validation
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
#algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
#evaluation
from sklearn import metrics



#unzip dataset and store in a temp dir
zip_ref = zipfile.ZipFile(config.path_to_zip_file_wine, 'r')
tmpdir = tempfile.mkdtemp()
print 'temporary directory at ' + tmpdir
zip_ref.extractall(tmpdir)
zip_ref.close()

#import dataset
file_name = tmpdir + '\winequality-red.csv'
data = pd.read_csv(file_name, sep = ';')
print 'The Red Wine Dataset opened'
#correlation
print data.corr()

#to get an initial impression on dataset
#to check for missing data in all columns
print(pd.isnull(data).any())
# visualization data
#bar chart
data['quality'] = pd.Categorical(data['quality'])
seaborn.countplot(x='quality', data=data)
plt.xlabel('Wine Quality Level ( scale: 0-10)')
print ('chart')
plt.show()

#divide features and target
X = data.drop('quality', axis=1) 
y = data.quality
#preprocessing (scale/normalizing)
X_scaled =  preprocessing.scale(X)

#7/3 split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.3)

#train model: random forest (parameter unrefined yet)
clf=RandomForestClassifier(n_estimators=40)
clf.fit(X_train, y_train)
#predict
y_pred = clf.predict(X_test)
#accuracy
accu = sklearn.metrics.accuracy_score(y_test, y_pred)
print('The accuracy score is :')
print accu
print ('The classification Report:')
print metrics.classification_report(y_test, y_pred)

#select parameter: 
n = 100
accuracy = [0]*n
confiInter = [0]*n

for i in range(n):
    clf = RandomForestClassifier(n_estimators=i+1)
    scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
    accuracy[i] = np.mean(scores)
    confiInter[i]= np.std(scores) *2
    

plt.plot(range(1, n+1), accuracy)
plt.xlabel("Number of trees")
plt.ylabel("Accuracy of prediction")
plt.title("Effect of the number of trees on the prediction accuracy")
plt.show()

# cv with random forest
clf = RandomForestClassifier(n_estimators=60)
scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
accuracy = np.mean(scores)
confiInter= np.std(scores) *2
print 'accuracy = ', accuracy, 'with confidence interval as', confiInter

#train model: SVM.SVC
#parameter unrefined
clf = svm.SVC(C=1,gamma=0.75,kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accu = sklearn.metrics.accuracy_score(y_test, y_pred)
print('The accuracy score is :')
print accu
print ('The classification Report:')
print metrics.classification_report(y_test, y_pred)

#select parameter
cArray = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
gArray = [0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5, 7.5, 10.0]
for i in cArray:
    for j in gArray:
        clf = svm.SVC(C=i, gamma=j, kernel='rbf')
        scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
        print 'C=', i, 'gamma=', j,'accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2
        
#from result hard to tell which one is better, c=0.1, 0.3, 1, gamma=0.1, 0.25 etc. for now only choose c=1, gamma=0.1
clf = svm.SVC(C=1, gamma=0.1, kernel='rbf')
scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
print 'accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2

#since both algorithms' performance less satisfying
#next step: select features
#most important feature for random forest on a 7/3 split:
#relative importance of each predictive variable:

forest = ExtraTreesClassifier(n_estimators=60)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

# Print the important features in order
print("Features ranking:")

for f in range(X_scaled.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

#use only chosen predictors
X = data[['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide']]
y = data.quality
#normalization
X_scaled =  preprocessing.scale(X)
#predict
clf = RandomForestClassifier(n_estimators=60)
scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
print 'accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2     

#select less features for svm
#because svc non-linear cannot select important feature like linearSVC does
#attempt select same features, use most correlated feature to quality
X = data[['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide', 'citric acid', 'density']]
#preprocessing (scale/normalizing)
X_scaled =  preprocessing.scale(X)
clf = svm.SVC(C=1, gamma=0.1, kernel='rbf')
scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
print 'accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2

#using polynomial features
#Random Forest
X = data[['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide']]
for i in range(2,5):
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    #normalization
    X_scaled =  preprocessing.scale(X_poly)
    #predict
    clf = RandomForestClassifier(n_estimators=60)
    scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
    print 'degree=', i, 'accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2
    
#using polynomial features
#SVC
X = data[['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide', 'citric acid', 'density']]
for i in range(2,5):
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    #normalization
    X_scaled =  preprocessing.scale(X_poly)
    #predict
    clf = svm.SVC(C=1, gamma=0.1, kernel='rbf')
    scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
    print ('degree =', i, 'accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2) 
    
#relcass dataset to low, medium, high quality
#low quality
data['quality']=data.quality.replace( [0, 4] , -1)
#medium quality
data['quality']=data.quality.replace( [5,6] , 0)
#high quality
data['quality']=data.quality.replace( [7, 10] , 1)

#for random forest, which need important features
X = data[['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide']] 
y = data.quality

#normalization
X_scaled =  preprocessing.scale(X)

#select parameter: 
n = 100 # this has to be 300 to see what happens
accuracy = [0]*n
confiInter = [0]*n

for i in range(n):
    clf = RandomForestClassifier(n_estimators=i+1)
    scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
    accuracy[i] = np.mean(scores)
    confiInter[i]= np.std(scores) *2

plt.plot(range(1, n+1), accuracy)
plt.xlabel("Number of trees")
plt.ylabel("Accuracy of prediction")
plt.title("Effect of the number of trees on the prediction accuracy")
plt.show()

#select parameters for svc
X = data[['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide', 'citric acid', 'density']]
#normalization
X_scaled =  preprocessing.scale(X)
#select parameter
cArray = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
gArray = [0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5, 7.5, 10.0]
for i in cArray:
    for j in gArray:
        clf = svm.SVC(C=i, gamma=0.1, kernel='rbf')
        scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
        print 'C=', i, 'gamma=', j,'accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2

#final prediction and evaluation
#forest
clf = RandomForestClassifier(n_estimators=80)
scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
accuracy = np.mean(scores)
confiInter= np.std(scores) *2
print 'accuracy = ', accuracy, 'with confidence interval as', confiInter

#bar chart with confidencial interval
f=pd.DataFrame(scores)
algo1=['forest']*10
f['algorithm'] = algo1

#svc
#with selected features
#X = data.drop('quality', axis=1)
X = data[['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide', 'citric acid', 'density']]
#normalization
X_scaled =  preprocessing.scale(X)
clf = svm.SVC(C=1, gamma=0.1, kernel='rbf')
scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=10)
print ('accuracy=',np.mean(scores), 'confidence interval = ', np.std(scores) *2)

#prepare to draw bar chart
svcScores=pd.DataFrame(scores)
algo2=['svc']*10
svcScores['algorithm'] = algo2
#add 2 dataFrame together
frames = [f, svcScores]
result = pd.concat(frames)
result.columns = ['accuScore', 'algorithm' ]

#draw accuracy and confidence interval of both algorithms
plt.xlabel(" wine quality prediction")
print ('chart')
finalBar=seaborn.barplot(x="algorithm", y="accuScore", data=result);

# Set these based on column counts for 2 bars (same width)
columncounts = [20,20]
#set unit bar width
widths = np.array(columncounts)/100.0

# Loop over the bars, and adjust the width (and position, to keep the bar centred)
for bar,newW in zip(finalBar.patches,widths):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.0
    bar.set_x(centre-newW/2.0)
    bar.set_width(newW)

plt.show()