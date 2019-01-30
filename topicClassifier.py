# @Author: Atul Sahay <atul>
# @Date:   2019-01-31T01:29:47+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2019-01-31T02:56:17+05:30



import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

#print("Done with import")
#To read data out of the test File
data = pd.read_csv("data/train_data.csv")
print(data.head())

#To sort the dataframe according to the id given
sorted_data = data.sort_values(by=['id'],ascending=True)
print(sorted_data.head())

sorted_data = sorted_data.reset_index(drop=True)
print(sorted_data.head(10))

#To read Test labels
test_labels = pd.read_csv("data/train_label.csv")
print(test_labels.head())

#To remove duplicate enteries
test_labels = test_labels.drop_duplicates(subset =['id'],\
                            keep = 'first')
print(test_labels.head(10))

#To get the total class labels
labels =  list(test_labels['label'].unique())
print(labels)

#To get map out of the list
labels = { k : v for v, k in enumerate(labels) }
print(labels)

#To change the strings with the mappings
test_labels = test_labels.replace({'label': labels})
print(test_labels.head(10))

#Need to join these class labels so that only one data frame remains
test_labels.index = sorted_data.index
cleaned_data = sorted_data.join(test_labels['label'])
print(cleaned_data.head(10))

#Now to do preprocessing
stemmer = SnowballStemmer('english')
words = stopwords.words("english")

cleaned_data['cleaned'] = cleaned_data['text'].apply( lambda x: \
                           " ".join([stemmer.stem(i) for i in \
                           re.sub("[^a-zA-Z]"," ",x).split() \
                           if i not in words]).lower())
print(cleaned_data['cleaned'].head(10))

#To split into the train and test samples to get the accuracy of the model
# We split the data set into the 20% test size
X_train, X_test, y_train, y_test = train_test_split(cleaned_data['cleaned'],\
                            cleaned_data['label'],test_size = 0.2)
# print(X_train.head())
# print(y_train.head())

# Now to build model from here
"""We will make use of the three things in the model fitting
first--- to make a vectorization or the feature set using the continuous bag
         of words by Tf idf vectorization, and too by bigram with ngram rnage(1,2)
second-- Need to extract top 10000 features from the gigantic matrix created by the
         tfidfvectorization
third--- by the selected features learn SVC through it, lasso regularization is used
         for better performance"""
