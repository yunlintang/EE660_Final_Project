import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



def prep(X):
    # drop attributes
    dropped = ['Veil-type','Stalk-root','Gill-color']
    X = X.drop(dropped,axis=1)
    
    # combine values
    X['Odor'] = X['Odor'].replace(['a','l'],['1','1'])
    X['Odor'] = X['Odor'].replace('c,f,m,p,s,y'.split(','),['2']*6)
    X['Stalk-color-above-ring'] = X['Stalk-color-above-ring'].replace(['e','g','o'],['1','1','1'])
    X['Stalk-color-above-ring'] = X['Stalk-color-above-ring'].replace(['b','c','n'],['2','2','2'])
    X['Stalk-color-below-ring'] = X['Stalk-color-below-ring'].replace(['e','g','o'],['1','1','1'])
    X['Stalk-color-below-ring'] = X['Stalk-color-below-ring'].replace(['b','c','n'],['2','2','2'])
    X['Ring-type'] = X['Ring-type'].replace(['l','n'],['1','1'])
    
    return X


def FeatEng(Xtrain,ytrain,Xtest,ytest):
    feats = ['Bruises', 'Cap-color', 'Cap-surface', 'Gill-attachment',
       'Gill-spacing', 'Habitat', 'Odor', 'Population', 'Ring-number',
       'Spore-print-color', 'Stalk-color-below-ring', 'Stalk-shape',
       'Stalk-surface-above-ring', 'Stalk-surface-below-ring',
       'Veil-color']
    Xtrain = prep(Xtrain)[feats]
    Xtest = prep(Xtest)[feats]
    
    oe = OrdinalEncoder()
    oe.fit(Xtrain)
    Xtrain = oe.transform(Xtrain)
    Xtest = oe.transform(Xtest)
    
    pca = PCA(n_components=8)
    pca.fit(Xtrain)
    Xtrain = pca.transform(Xtrain)
    Xtest = pca.transform(Xtest)
    
    le = LabelEncoder()
    le.fit(ytrain)
    ytrain = le.transform(ytrain)
    ytest = le.transform(ytest)
    
    return Xtrain,ytrain,Xtest,ytest


def preprocess_data():
  # read data
  feats = "Cap-shape, Cap-surface, Cap-color, Bruises, Odor, Gill-attachment, Gill-spacing, Gill-size, Gill-color, Stalk-shape, Stalk-root, Stalk-surface-above-ring, Stalk-surface-below-ring, Stalk-color-above-ring, Stalk-color-below-ring, Veil-type, Veil-color, Ring-number, Ring-type, Spore-print-color, Population, Habitat, Class"
  cols = [x.strip() for x in feats.split(',')]
  df = pd.read_csv('data/mushroom.csv',header=None, names=cols)

  # split dataset
  X, X_test, y, y_test = train_test_split(df.iloc[:,:-1], df['Class'], test_size=0.2, random_state=1)

  # split for pre-training
  X_train, X_pre, y_train, y_pre = train_test_split(X,y,test_size=0.2,random_state=0)

  # feature engineering on data
  X_train, y_train, X_test, y_test = FeatEng(X_train, y_train, X_test, y_test)

  # unlabel data, convert to SSL dataset
  # for SL methods
  X_train_l, X_train_u, y_train_l, y_train_u = train_test_split(X_train,y_train,test_size=0.7,random_state=1)
  # for SSL methods
  X_train_ssl = np.concatenate((X_train_l,X_train_u))
  y_train_ssl = np.concatenate((y_train_l,np.full(y_train_u.shape,-1)))

  # save data
  pd.DataFrame(X_train_l).to_csv('data/X_train_l.csv',index=False)
  pd.DataFrame(X_train_u).to_csv('data/X_train_u.csv',index=False)
  pd.DataFrame(y_train_l).to_csv('data/y_train_l.csv',index=False)
  pd.DataFrame(y_train_u).to_csv('data/y_train_u.csv',index=False)

  pd.DataFrame(X_train_ssl).to_csv('data/X_train_ssl.csv',index=False)
  pd.DataFrame(y_train_ssl).to_csv('data/y_train_ssl.csv',index=False)

  pd.DataFrame(X_test).to_csv('data/X_test.csv',index=False)
  pd.DataFrame(y_test).to_csv('data/y_test.csv',index=False)

  return