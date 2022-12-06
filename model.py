import pandas as pd
import numpy as np
import random

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelPropagation
from qns3vm.qns3vm import QN_S3VM
from sklearn.metrics import f1_score

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# function for evaluating the SL models
# score function: f1-score
def cross_val_sl(X,y,clf):
    scores = cross_validate(clf,X,y,scoring='f1')
    print('avg. of val scores: {}'.format(round(np.mean(scores['test_score']),3)))
    print('std. of val scores: {}'.format(round(np.std(scores['test_score']),3)))
    return (scores['test_score'])


# function for evaluating the SSL by using KFold (k=5)
# score function: f1-score
def cross_val_ssl(X,y,clf):
    scores = []
    kfold = KFold(n_splits=5,shuffle=True,random_state=1)
    for train_idx, test_idx in kfold.split(X,y):
        clf.fit(X[train_idx],y[train_idx])
        val_idx = y[test_idx] != -1
        pred = clf.predict(X[test_idx][val_idx])
        scores += [f1_score(pred,y[test_idx][val_idx])]
    print('avg. of val scores: {}'.format(round(np.mean(scores),3)))
    print('std. of val scores: {}'.format(round(np.std(scores),3)))
    return (scores)


def read_data():
  X_train_l = pd.read_csv('data/X_train_l.csv')
  X_train_u = pd.read_csv('data/X_train_u.csv')
  y_train_l = pd.read_csv('data/y_train_l.csv')
  y_train_u = pd.read_csv('data/y_train_u.csv')

  X_train_ssl = pd.read_csv('data/X_train_ssl.csv')
  y_train_ssl = pd.read_csv('data/y_train_ssl.csv')

  X_test = pd.read_csv('data/X_test.csv')
  y_test = pd.read_csv('data/y_test.csv')

  return X_train_l, X_train_u, y_train_l, y_train_u,X_train_ssl,y_train_ssl,X_test,y_test


def model_training():
  X_train_l, X_train_u, y_train_l, y_train_u,X_train_ssl,y_train_ssl,X_test,y_test = read_data()

  # SL
  y_train_l = y_train_l.to_numpy().reshape(len(y_train_l))
  # baseline trivial (predict based on prob. of train)
  print("\nSL-trivial baseline")
  clf_dum = DummyClassifier(strategy='stratified',random_state=2)
  cross_val_sl(X_train_l,y_train_l,clf_dum)
  clf_dum.fit(X_train_l,y_train_l)

  # baseline non-trivial (logistic regression)
  print('\nSL-baseline logistic regression')
  clf_log = LogisticRegression(random_state=0)
  cross_val_sl(X_train_l,y_train_l,clf_log)
  clf_log.fit(X_train_l,y_train_l)

  # decison tree
  # hyper-param. tuning for decision tree
  print('\nSL-decision tree')
  for d in range(1,6):
      print('max_depth={}'.format(d))
      clf = DecisionTreeClassifier(max_depth=d,random_state=24)
      cross_val_sl(X_train_l,y_train_l,clf)

  # decision tree classifier
  clf_tree = DecisionTreeClassifier(max_depth=4,random_state=10)
  cross_val_sl(X_train_l,y_train_l,clf_tree)
  clf_tree.fit(X_train_l,y_train_l)

  # linear SVM classifier
  print('\nSL-SVM')
  # hyper-param. tuning for SVM
  for c in [0.01,0.05,0.1,0.5,1]:
      print('lambda={}'.format(c))
      clf = SVC(kernel='linear',C=c)
      cross_val_sl(X_train_l,y_train_l,clf)

  clf_svm = SVC(kernel='linear',random_state=10)
  cross_val_sl(X_train_l,y_train_l,clf_svm)
  clf_svm.fit(X_train_l,y_train_l)


  # SSL
  X_train_ssl = X_train_ssl.to_numpy()
  y_train_ssl = y_train_ssl.to_numpy().reshape(len(y_train_ssl))
  # self-training model
  print('\nSSL-self training')
  clf_self = SelfTrainingClassifier(base_estimator=DecisionTreeClassifier(max_depth=4,random_state=24),max_iter=1000)
  cross_val_ssl(X_train_ssl,y_train_ssl,clf_self)
  clf_self.fit(X_train_ssl,y_train_ssl)

  # S3VM
  print('\nSSL-S3VM')
  # provided by NekoYIQI's Github Repo.
  # https://github.com/NekoYIQI/QNS3VM

  warnings.filterwarnings('ignore')
  # convert {0,1} to {-1,1}
  y_trtemp_l = y_train_l.copy()
  y_trtemp_l[y_trtemp_l==0] = -1

  # random generator
  myRandom = random.Random()
  myRandom.seed(0)

  scores = []
  for l in [0.01,0.05,0.1,0.5,1]:
      # train model
      clf = QN_S3VM(X_l=X_train_l.to_numpy().tolist(),L_l=y_trtemp_l.tolist(),
                        X_u=X_train_u.to_numpy().tolist(),random_generator=myRandom,kernel_type='Linear',lam=l)
      clf.train()
      pred = clf.getPredictions(X=X_train_l)
      score = f1_score(pred,y_trtemp_l)
      print("Traning Scores with lambda = {}: {}".format(l,score))
      
      scores += [score]

  # label propagation
  print('\nSSL-label propagation')
  clf_har_1 = LabelPropagation(kernel='knn')
  cross_val_ssl(X_train_ssl,y_train_ssl,clf_har_1)

  clf_har_2 = LabelPropagation(kernel='rbf')
  cross_val_ssl(X_train_ssl,y_train_ssl,clf_har_2)

  clf_har = LabelPropagation(kernel='rbf')
  clf_har.fit(X_train_ssl,y_train_ssl)


def final_model_training():
  # function for only training the final system
  X_train_l, X_train_u, y_train_l, y_train_u,X_train_ssl,y_train_ssl,X_test,y_test = read_data()

  # decision tree classifier
  clf_tree = DecisionTreeClassifier(max_depth=4,random_state=10)
  clf_tree.fit(X_train_l,y_train_l)
  score_dt = np.round(f1_score(clf_tree.predict(X_test),y_test),4)
  print('test f1-score for decision tree: {}'.format(score_dt))

  # label propagation classifier
  clf_har = LabelPropagation(kernel='rbf')
  y_train_ssl = y_train_ssl.to_numpy().reshape(len(y_train_ssl))
  clf_har.fit(X_train_ssl,y_train_ssl)
  score_lp = np.round(f1_score(clf_har.predict(X_test),y_test),4)
  print('test f1-score for label propagation: {}'.format(score_lp))

  return