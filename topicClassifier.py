# @Author: Atul Sahay <atul>
# @Date:   2019-01-31T01:29:47+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2019-01-31T04:45:03+05:30



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
class_labels =  list(test_labels['label'].unique())
print(class_labels)

#To get map out of the list
labels = { k : v for v, k in enumerate(class_labels) }
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

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range = (1,2), stop_words=\
                            "english", sublinear_tf = True)), \
                     ('chi', SelectKBest(chi2, k=10000)), \
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter = 3000, \
                            dual=False))])
model = pipeline.fit(X_train,y_train)

# To extract top fetaures for each class
# its just for the visualization

# Need to search more about it
vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']
#------------------------------

feature_names = vectorizer.get_feature_names()
feature_names = [ feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)
print(feature_names[:10])

#Top 10 features or the keywords
target_names = [ str(i) for i in range(15)]
print("top 10 keywords per class")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print(" %s : %s "%(label," ".join(feature_names[top10])))

print("accuracy score: "+str(model.score(X_test,y_test)))
print(model.predict(["These machine screw nuts are designed to be used with smaller machine screws (under 1/4 in.) and have a hex drive. Used for fastening to a screw when mechanically joining materials together. Must be used with like materials/sized screws. Available in various materials and finishes to suit your application.California residents: see&nbsp"]))


# Now to load test data set and get the final readings
Test_Set = pd.read_csv("data/test_data.csv")
Test_labels = model.predict(Test_Set['text'])
print(Test_labels[:10])

mapping = { k: v for k,v in enumerate(class_labels) }
print(mapping)

submission_data = pd.read_csv("data/sample_submission.csv")
print(submission_data.head(10))

for i in range(len(Test_labels)):
    submission_data.at[i,mapping[Test_labels[i]]] = 1

print(submission_data.head(10))

submission_data.to_csv("FinalSubmission.csv", encoding='utf-8', index=False)
