
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import lda
from sklearn.feature_extraction.text import CountVectorizer

debug = 0

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
  if debug == 1:
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

def documentFrequencyVector(trainData):
  tfv = CountVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), stop_words='english')
  tfv.fit(trainData)
  X = tfv.transform(trainData)
  return X

def getLdaTopicsVector(X):
  model = lda.LDA(n_topics=20, n_iter=1, random_state=1)
  model.fit(X)  # model.fit_transform(X) is also available
  doc_topic = model.doc_topic_
  topicVector = []
  for i in range(X.shape[0]):
    topicVector.append(doc_topic[i].argmax())
  return topicVector

def transformInLDATopics(X):
  model = lda.LDA(n_topics=1000, n_iter=10, random_state=1)
  model.fit(X)  # model.fit_transform(X) is also available
  return model.doc_topic_

def selectModel(classifier):
  if classifier == 0:
    print 'Model: Logistic Regression'
    model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
  elif classifier == 1:
    print 'Model: Naive Bayes'
    model = MultinomialNB(alpha=1)
  elif classifier == 2:
    print 'Model: Random Forest'
    #model = RandomForestClassifier(n_estimators=10, max_features="auto", max_depth=2, bootstrap=True, n_jobs=4)
    model = RandomForestClassifier(n_estimators=10, max_features="auto", max_depth=None, bootstrap=True, n_jobs=4, criterion="entropy")
    #model = RandomForestClassifier(n_estimators=200, max_features=None, max_depth=None, bootstrap=True, n_jobs=-1, verbose=1)
    #model = RandomForestClassifier(n_estimators=10)
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

def plotFeatureImportances(forest):
  importances = forest.feature_importances_
  std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
  indices = np.argsort(importances)[::-1]

  # Print the feature ranking
  print("Feature ranking:")

  for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

  # Plot the feature importances of the forest
  plt.figure()
  plt.title("Feature importances")
  plt.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center")
  plt.xticks(range(10), indices)
  plt.xlim([-1, 10])
  plt.show()

def egon(rawData, trainFile, testFile, classifier, boilerplateOnly):
  # Feature Engineering and Formatting Data
  trainData = featureEng(trainFile)
  if debug == 1:
    print "trainData shape:", trainData.shape
  # For Submission
  #testData = featureEngTest(testFile)

  # TFIDF with LR on boilerplate
  trainDataBoilerplate = list(np.array(p.read_table('train.tsv'))[:,2])
  testDataBoilerplate = list(np.array(p.read_table('test.tsv'))[:,2])
  
  y = list(np.array(p.read_table('train.tsv'))[:,-1])
  target = np.array(p.read_table('train.tsv'))[:,-1]
  isImbalanced(y)

  if boilerplateOnly == 0:
    print "Feature selection type: TFIDF on boiler plate"
    X = tfidf(trainDataBoilerplate)
  elif boilerplateOnly == 1:
    print "Feature selection type: Non boiler plate features"
    X = trainData
  elif boilerplateOnly == 2:
    print "Feature selection type: Non boiler plate features + topics extracted using LDA"
    X = documentFrequencyVector(trainDataBoilerplate)
    topicsVector = getLdaTopicsVector(X)
    t = np.array(topicsVector)[np.newaxis].T
    trainData = np.append(trainData, t, 1)
    X = trainData
  elif boilerplateOnly == 3:
    print "Feature selection type: topics extracted using LDA"
    X = documentFrequencyVector(trainDataBoilerplate)
    X = transformInLDATopics(X)
  else:
    print "boilerplateOnly value should either be 0, 1, 2 or 3"
    sys.exit(0)

  model = selectModel(classifier)

  # Feature importances

  if debug == 1:
    print "===== Starting random split ====="
  aucAvg = 0.0
  numcross = 10
  for i in xrange(numcross): 
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size = 0.2)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:,1]
    auc = metrics.roc_auc_score(y_cv, preds)
    if debug == 1:
      print "NumCross:",i," Auc:",auc
    aucAvg += auc
  aucAvg = aucAvg/numcross
  print numcross,"Random split AUC : %f" % aucAvg
 
  #plotFeatureImportances(model)

  if debug == 1:
    print "===== Starting k fold validation ====="
  aucAvg = 0.0
  folds = 10
  cv = cross_validation.KFold(len(trainData), n_folds=folds)
  fold = 0
  for traincv, testcv in cv:
    X_train, X_cv, y_train, y_cv = X[traincv], X[testcv], target[traincv], target[testcv]
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:,1]
    fold = fold + 1
    auc = metrics.roc_auc_score(y_cv, preds)
    if debug == 1:
      print "Fold:",fold," Auc:",auc
    aucAvg += auc
  aucAvg = aucAvg/folds
  print fold,"fold AUC : %f" % aucAvg
  
  '''
  # For Submission
  model.fit(X, y)
  preds = model.predict_proba(testData)[:,1]
  testfile = p.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
  pred_df = p.DataFrame(preds, index=testfile.index, columns=['label'])
  pred_df.to_csv('benchmark-kewal.csv')
  '''

def main():
  global debug
  if len(sys.argv) < 6:
    print 'Usage: python main.py <rawData> <trainFile> <testFile> <classifier - 0-lr, 1-nb, 2-rf> <feature selection 0-boilerplateOnly, 1-useOtherFeatures, 2-userOtherFeatures + LDA, 3-LDA>'
    sys.exit(2)
  if len(sys.argv) == 7:
    debug = int(sys.argv[6])
  egon(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])) 

main()
