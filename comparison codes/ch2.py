from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer	
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_selection import VarianceThreshold
import logging
from optparse import OptionParser
import sys
from time import time
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
import random
import scipy

from scipy import sparse,io
categories = ['soc.religion.christian','comp.graphics']


y_train=np.loadtxt("ytrain.txt")
y_test=np.loadtxt("ytest.txt")

true_k = np.unique(y_train).shape[0]
X_train=io.mmread("xtrain.mtx").tocsr()
X_test=io.mmread("xtest.mtx").tocsr()

ch2 = SelectKBest(chi2)
X_train1 = ch2.fit_transform(X_train, y_train)
X_test1 = ch2.transform(X_test)
print X_train1.shape
print X_train.shape

clf = MultinomialNB(alpha=.01)
clf.fit(X_train1, y_train)
pred = clf.predict(X_test1)


print(metrics.classification_report(y_test, pred,target_names=categories))

