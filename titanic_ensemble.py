#importing the library
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset_train= dataset_train.drop(dataset_train.columns[[0,3,5,6,7,8,10]], axis=1)
dataset_test= dataset_test.drop(dataset_test.columns[[0,2,4,5,6,7,9]], axis=1)

#finding out rows having null data in specific columns in training set
cols_train = list(dataset_train)
null_rows_train = pd.DataFrame()
for i in cols_train:
    null_rows_train = null_rows_train.append(dataset_train[dataset_train[i].isnull()])    
    
#finding out rows having null data in specific columns in test set
cols_test = list(dataset_test)
null_rows_test = pd.DataFrame()
for i in cols_test:
    null_rows_test = null_rows_test.append(dataset_test[dataset_test[i].isnull()])
  
#updating the missing values with random embarkment (S,C,Q) in training set
import random
embark = ['S','C','Q']
dataset_train['Embarked'] = dataset_train['Embarked'].fillna(random.choice(embark))

#Assining the dataset values 
X = dataset_train.iloc[:, 1:5].values
y = dataset_train.iloc[:, 0].values
X_test = dataset_test.values

#updating the missing values in testing set 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, [2]])
X_test[:, [2]] = imputer.transform(X_test[:, [2]])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_gender = LabelEncoder()
labelencoder_embarked = LabelEncoder()
X[:, 1] = labelencoder_gender.fit_transform(X[:, 1])
X_test[:,1] = labelencoder_gender.transform(X_test[:, 1])
X[:, 3] = labelencoder_embarked.fit_transform(X[:, 3])
X_test[:, 3] = labelencoder_embarked.transform(X_test[:, 3])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

#importing necessary libraries for implementing the classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Initializing different classifiers
clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = RandomForestClassifier(n_estimators = 100,random_state=1)
clf3 = GaussianNB()
clf4 = SVC(kernel='rbf')
lr = LogisticRegression(solver='liblinear')

#Fitting different classifiers to the training set
clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)

#predicting using KNN
f1 = clf1.predict(X_test)

#predicting using Random Forest
f2 = clf2.predict(X_test)

#predicting using Naive bayes
f3 = clf3.predict(X_test)

#predicting using Kernel SVM
f4 = clf4.predict(X_test)

#Initialising stacking classifier
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4], meta_classifier=lr)
sclf.fit(X,y)
final_prediction = sclf.predict(X_test)
