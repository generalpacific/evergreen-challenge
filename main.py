
import numpy as np
import scipy
import sys
import math
from random import sample
import time
import pandas as p
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
SEED = 15

def convert(val):
  if val == '?':
    return 0
  elif isinstance(val,basestring):
    return float(val)
  else:
    return val

def isImbalanced(y):
  class0=0.0
  class1=0.0
  for e in y:
    if e == 0:
      class0+=1
    else:
      class1+=1
  print 'Class 0:', class0, 'Class 1:', class1
  if (max(class0,class1)/min(class0,class0) > 3):
    print 'Imbalanced Classes'
  else:
    print 'No Imbalanced Classes'

def tfidf(trainData):
  tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
  tfv.fit(trainData)
  X = tfv.transform(trainData)
  return X

def selectModel(classifier):
  if classifier == 0:
    print 'Model: Logistic Regression'
    model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
  elif classifier == 1:
    print 'Model: Naive Bayes'
    model = MultinomialNB(alpha=1)
  return model

def featureEng(trainFile):
  trainData = np.array(p.read_table(trainFile, converters={'alchemy_category_score':convert, 'is_news':convert, 'news_front_page':convert}))
  # Delete unnecessary features
  # Delete URL, URLID, BolierPlate, Alchemy_Category, Label
  #trainData = np.delete(trainData, [0,1,2,3,11,12,14,-1], 1)
  trainData = np.delete(trainData, [0,1,2,3,11,12,14,21,26], 1)

  # This fixed the attribute error(while predicting)
  trainData = np.array(trainData, dtype=np.float)
  return trainData

def featureEngTest(testFile):
  testData = np.array(p.read_table(testFile, converters={'alchemy_category_score':convert, 'is_news':convert, 'news_front_page':convert}))
  # Delete unnecessary features
  # Delete URL, URLID, BolierPlate, Alchemy_Category, Label
  #trainData = np.delete(trainData, [0,1,2,3,11,12,14,-1], 1)
  testData = np.delete(testData, [0,1,2,3,11,12,14,21], 1)

  # This fixed the attribute error(while predicting)
  testData = np.array(testData, dtype=np.float)
  return testData

def egon(rawData, trainFile, testFile, classifier, boilerplateOnly):
  # Feature Engineering and Formatting Data
  trainData = featureEng(trainFile)
  # For Submission
  #testData = featureEngTest(testFile)

  # TFIDF with LR on boilerplate
  trainDataBoilerplate = list(np.array(p.read_table('train.tsv'))[:,2])
  testDataBoilerplate = list(np.array(p.read_table('test.tsv'))[:,2])
  
  y = list(np.array(p.read_table('train.tsv'))[:,-1])
  isImbalanced(y)

  if boilerplateOnly == 1:
    X = tfidf(trainDataBoilerplate)
  else:
    X = list(trainData) 

  model = selectModel(classifier)
  aucAvg = 0.0
  for i in xrange(10): 
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size = 0.2, random_state = 1*SEED)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:,1]
    auc = metrics.roc_auc_score(y_cv, preds)
    aucAvg += auc
  aucAvg = aucAvg/10
  print "AUC : %f" % aucAvg

  # For Submission
  #model.fit(X, y)
  #preds = model.predict_proba(testData)[:,1]
  #testfile = p.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
  #pred_df = p.DataFrame(preds, index=testfile.index, columns=['label'])
  #pred_df.to_csv('benchmark-kewal.csv')

def main():
  if len(sys.argv) < 6:
    print 'Usage: python cr.py <rawData> <trainFile> <testFile> <classifier - 0-lr, 1-nb> <boilerplateOnly>'
    sys.exit(2)
  egon(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])) 

main()
