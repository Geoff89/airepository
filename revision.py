# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:39:15 2017

@author: jeffnerd
"""


iris = load_iris()
X = iris.data # 
y = iris.target # one d array dimension response

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 0)


#simulation for the kfolds
from sklearn.cross_validation import KFold
kf = KFold(25,n_folds = 5, shuffle = False)
#print the contents for each traiing and testing set
print('{0},{1},{2}'.format('Iteration','Training Set Observations','Testing Set Observations'))
for Iteration,data in enumerate(kf,start = 1):
    print('{0},{1},{2}'.format(Iteration,data[0],data[1]))
    
Iteration,Training Set Observations,Testing Set Observations
1,[ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24],[0 1 2 3 4]
2,[ 0  1  2  3  4 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24],[5 6 7 8 9]
3,[ 0  1  2  3  4  5  6  7  8  9 15 16 17 18 19 20 21 22 23 24],[10 11 12 13 14]
4,[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 20 21 22 23 24],[15 16 17 18 19]
5,[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19],[20 21 22 23 24]    
#Everr onbservation is in the testing set exacty once

from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print(scores)
[ 1.          0.93333333  1.          1.          0.86666667  0.93333333
  0.93333333  1.          1.          1.        ]
print(scores.mean())
0.966666666667

#Looping
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
    print(k_scores)
    
# plotting helps us choose the best model of K to use as K
# Higher values of K produce thesimplest mode which has a lower complexity
#Nb Lower values of K have a loweer bias and higer variance and higher values
#of K have a higher biase and lower variance
# models in the middle are the best as they balance the bias-variance trade off

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross validation Accuracy')


#choosing the best model for a given assignment
#10 folds cross validation with the best KNN mode
knn =  knn = KNeighborsClassifier(n_neighbors = 20)
scores = cross_val_score(knn,X,y,cv=10, scoring = 'accuracy')
print(scores.mean())
0.98
#10 folds cross validatipn for logistic regression
logreg = LogisticRegression()
scores = cross_val_score(logreg,X,y,cv=10, scoring = 'accuracy')
print(scores.mean())
0.98



    
    
    
    
                             