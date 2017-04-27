from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import sparse,io
import time
from scipy.sparse import vstack
import scipy
from collections import OrderedDict
import operator
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import classification_report

def Classification(X_train, X_test, y_train, y_test):
	print "\n----------SVM linear Kernel--------\n"
	start1 = time.time()
	clf = svm.SVC(kernel='linear', C=1)
	clf.fit(X_train, y_train)
	print "Training time taken %f \n" % (time.time() - start1)

	start2 = time.time()
	y_pred = clf.predict(X_test).tolist()
	print "Testing time taken %f \n" % (time.time() - start2)
	print "Classification report:\n"
	print metrics.classification_report(y_test, y_pred)


#Function to delete rows from a csr matrix
def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

start=time.clock()
categories = ['soc.religion.christian','comp.graphics']

#y_train = doc vs class
y_train=np.loadtxt("ytrain.txt")
y_test=np.loadtxt("ytest.txt")

#X_train = doc vs term
X_train=io.mmread("xtrain.mtx").tocsr()
X_test=io.mmread("xtest.mtx").tocsr()

class_mat=X_train.todense()
mat=class_mat.tolist()
t_d=class_mat.T.tolist()

ytrain=y_train.tolist()

#--------------------Clustering begins------------------------------

#nodocs, noterms are number of documents and terms respectively belonging to all clusters
nodocs,noterms=X_train.shape

#sum1, sum2 = average sum of tf-idfs for each term in class1 and class2 respectively
sum1=[0 for x in range(noterms)]
sum2=[0 for x in range(noterms)]

#n_1, n_2 are the number of documents in class1 and class2 respectively
n_1=0
n_2=0

for i in range(len(ytrain)):
	if ytrain[i]==0.0:
		n_1=n_1+1
	else:
		n_2=n_2+1

for i in range(noterms):

	for j in range(nodocs):
		if ytrain[j]==0:
			sum1[i]=sum1[i]+t_d[i][j]

		else:
			sum2[i]=sum2[i]+t_d[i][j]

	sum2[i]=sum2[i]/n_2
	sum1[i]=sum1[i]/n_1
#print len(sum1), len(sum2)

#cosim_1 is the cosine similarity of each document w.r.t to the centroid document of its respective class.
cosim_1=[0 for x in range(nodocs)]

#Used for calculate average cosine similarity
sum_1=0
sum_2=0
for i in range(nodocs):
	if ytrain[i]==0:
		cosim_1[i]=cosine_similarity(sum1,mat[i])[0][0]

		sum_1=sum_1+cosim_1[i]
	else:
		cosim_1[i]=cosine_similarity(sum2,mat[i])[0][0]

		sum_2=sum_2+cosim_1[i]

thres_1=sum_1/n_1


thres_2=sum_2/n_2

#docs is the set of documents which lie far away from the centroid. These must be discarded
docs=set()
i=nodocs
while i>0:
	i=i-1

	if ytrain[i]==0 and cosim_1[i]<0.2:
		docs.add(i)

	elif ytrain[i]==1 and cosim_1[i]<0.007:
		docs.add(i)


dd=list(docs)
i=nodocs
while i>0:
	i=i-1
	if i not in docs:
		continue
	else:
		#deleting from X_train and y_train
		delete_row_csr(X_train,i)
		y_train=np.delete(y_train,i,0)
#=============================Clustering ends=======================================

#-----------------------------Applying variance threshold---------------------------
selector = VarianceThreshold(0.000005)
X_train1=selector.fit_transform(X_train)
X_test1=selector.transform(X_test)

#X_train1=X_train
#X_test1=X_test

#X_train1 contains the set of new documents
#=============================Variance ends==========================================
nodocs,noterms=X_train1.shape

tfidfl=X_train1.todense().tolist()

#----------------------------DPM calculation begins----------------------------------

#n_1 and n_2 are the new number of documents in class1 and class2 respectively.
n_1=0
n_2=0
ytrain=y_train.tolist()

for i in range(len(ytrain)):
	if ytrain[i]==0.0:
		n_1=n_1+1
	else:
		n_2=n_2+1

#dfi is the document frequency inside each category
dfi = [[0 for i in range(noterms)] for j in range(2)]

#dfo is the document frequency outside each category
dfo = [[0 for i in range(noterms)] for j in range(2)]

for i in range(noterms):
	for j in range(nodocs):
		if tfidfl[j][i]!=0:
			if ytrain[j]==0:
				dfi[0][i]=dfi[0][i]+1
				dfo[1][i]=dfo[1][i]+1
			else :
				dfi[1][i]=dfi[1][i]+1
				dfo[0][i]=dfo[0][i]+1


for i in range(noterms):
	dfo[0][i]=float(dfo[0][i])/n_2
	dfo[1][i]=float(dfo[1][i])/n_1
	dfi[1][i]=float(dfi[1][i])/n_2
	dfi[0][i]=float(dfi[0][i])/n_1

#delta = absolute difference of DFi and DFo for each category and each term
delta = [[0 for i in range(noterms)] for j in range(2)]
for i in range(noterms):
	delta[0][i]=abs(dfi[0][i]-dfo[0][i])
	delta[1][i]=abs(dfi[1][i]-dfo[1][i])
	
#dpm array stores the DPM for each term
dpm = [0 for j in range(noterms)]

for i in range(noterms):
	dpm[i]=float(delta[0][i]+delta[1][i])

#============================DPM ends===========================================

					
tfid_list=X_train1.todense().tolist()

docu,ter=X_train1.shape

tvsd=X_train1.todense().T.tolist() #term vs document

#loading the correlation matrix stored in a file.
rum=np.load("plz.txt.npy")
y=rum.tolist()
corr=[[10.0 for i in range(ter)] for j in range(ter)]

for i in range(ter):
	for j in range(ter):
		if corr[i][j]==10.0:
			corr[i][j]=y[i][j]
			#corr[j][i]=corr[i][j]

#Boolean array to determine whether to access the term-row
boole=[0 for i in range(ter)]#0-true

#Creating a set of terms need to be chosen
chosenones=set()

cont=[]
count = 0
for i in range(ter):
	if boole[i]==0:
		dicti={}
		for j in range(ter):
			if (corr[i][j]>=0.7 or corr[i][j]<=-0.7 ):
				dicti[j]=dpm[j]
				
				#chosenones.add(j)
		#if len(dicti)>5:
			#cont.append(len(dicti))
		d_descending = OrderedDict(sorted(dicti.items(), key=operator.itemgetter(1),reverse=True))
		if len(d_descending)<6:
			for c in range(len(d_descending)):
				boole[d_descending.keys()[c]]=1
				chosenones.add(d_descending.keys()[c])
		else:
			for c in range(6):
				boole[d_descending.keys()[c]]=1
				chosenones.add(d_descending.keys()[c])

m=X_train1.todense().T		
m1=X_test1.todense().T


g=[i for i in range(ter)]
#ad = set of all terms
ad=set(g)

#z = set of all terms not used
z=chosenones
#z=(sorted(ad-chosenones))
l=list(z)

#Removing the terms from the training and testing set
i=ter
while i>=0:
	i=i-1
	if i in z:
		m=np.delete(m,i,0)
		m1=np.delete(m1,i,0)
	else: 
		continue

X_train2=sparse.csr_matrix(m).T
X_test2=sparse.csr_matrix(m1).T



#clf = MultinomialNB(alpha=.01)
#clf.fit(X_train2, y_train)
#pred = clf.predict(X_test2)

Classification(X_train2,X_test2,y_train,y_test)
#print(metrics.classification_report(y_test, pred,target_names=categories))


