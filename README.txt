IR PROJECT 
----------------------
GROUP MEMBERS:
1)Sukriti Tiwari 2013A7PS086G
2)Srishti Sharma 2013A7PS106G
3)Rumana Lakdawala 2013A7PS113G
4)Siddharth Mohan 2013A7PS276G
5)RAM ALIAS GAURAV GIRISH TAMBA 2012B1A7327G 	


Approach
---------------
1)Cluster Documents based on classwise centroid to get a superior subset of documents.
2)Apply a variance threshold(0.00001) to eliminate terms whose tf-idf remains roughly the same throughout.
3)Calculate discriminating power measure (dpm) of each term
4)Based on correlation of terms, find a subset of terms decided by correlation and dpm.


JUSTIFICATION
----------------------
1) While clustering the documents, we only select those whose cosine similarity lies within a threshold of the cosine similarity of the class wise centroid.By doing this, we eliminate those documents which vary greatly from the centroid, and lie at the border of the class 1 , class 0 intersection.

2) By  applying variance threshold, we are removing the terms whose tf-idf varies very less over all the documents. (example: for term 1 ,tf-idf over corpus of three documents is 1.43,1.431,1.428)
Since these terms vary less, they are not giving us much information. 
Hence these are eliminated. 
Number of terms eliminated at this stage are less(since variance threshold kept very low)

*****************************************************************
Genari et al. [GLF89] state that
“Features are relevant if their values vary systematically with category membership.”

In other words, a feature is useful if it is correlated with or predictive of the class; otherwise
it is irrelevant. Kohavi and John [KJ96] formalize this definition as
Definition 1: A feature Vi is said to be relevant iff there exists some vi and c for which
p(Vi = vi) > 0 such that p(C = c|Vi = vi) 6= p(C = c).

Empirical evidence from the feature selection literature shows that, along with irrelevant
features, redundant information should be eliminated as well [LS94a, KJ96, KS95].
 
A feature is said to be redundant if one or more of the other features are highly correlated with it. The above definitions for relevance and redundancy lead to the following hypothesis,
on which the feature selection method presented in this thesis is based:
A good feature subset is one that contains features highly correlated with
(predictive of) the class, yet uncorrelated with (not predictive of) each other.

**********************************************************************


3)Next, we find correlation between all terms, based on these, we find a subset of highly correlated terms for each term, now for these we choose the top k based on discriminating power measure(ie, how well the term can contribute to differentiating(dicriminating) between the class labels)

Highly correlated terms(>0.8) will vary similarly, ie. as one varies, so will the other by some fixed function. These, hence convey similar information.
Now by choosing the top k, we are effectively choosing the terms whose information is "useful".
Where useful means that they contribute to discriminate between the classes.

4)Now we apply a  classifier to this data.