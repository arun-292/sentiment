#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# In[3]:


temp = pd.read_csv('1429_1.csv')


# In[4]:


temp.head()


# In[5]:


# In[6]:


permanent = temp[['reviews.rating','reviews.text','reviews.title','reviews.username']]
permanent.head()





# In[9]:


check = permanent[permanent['reviews.rating'].isnull()]
check.head()


# In[10]:


senti = permanent[permanent['reviews.rating'].notnull()]
permanent.head()







# In[13]:


#for i in last_df['reviews.rating']:
    #j=1;
    #if i >= 4 :
      #  last_df['type'][j] = "Positive"
       # j=j+1
        
    #else:
     #   last_df['type'][j] = 'Negative'
      #  j=j+1
   
    


# In[ ]:





# In[14]:


senti['senti'] = senti['reviews.rating']>=4
senti['senti'] = senti['senti'].replace([True, False] , ['pos', 'neg'])





# In[16]:


senti['senti'].value_counts().plot.bar()




# In[18]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[19]:


cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ',sentence).strip()
    return sentence


# In[20]:


senti['Summary_Clean'] = senti['reviews.text'].apply(cleanup)
check['Summary_Clean'] = check['reviews.text'].apply(cleanup)


# In[21]:


split = senti[["Summary_Clean", "senti"]]
train = split.sample(frac=0.8,random_state=200)
test=split.drop(train.index)


# In[22]:


def word_feats(words):
    features = {}
    for word in words:
        features[word] = True
    return features


# In[ ]:





# In[23]:


train


# In[24]:


test


# In[25]:


train['words'] =train['Summary_Clean'].str.lower().str.split()
test['words'] =test['Summary_Clean'].str.lower().str.split()
check['words'] =check['Summary_Clean'].str.lower().str.split()

train.index =range(train.shape[0])
test.index = range(test.shape[0])
check.index = range(check.shape[0])

prediction = {}
train_naive = []
test_naive = []
check_naive =[]

for i in range(train.shape[0]):
    train_naive = train_naive +[[word_feats(train["words"][i]) , train["senti"][i]]]
for i in range(test.shape[0]):
    test_naive = test_naive +[[word_feats(test["words"][i]) , test["senti"][i]]]
for i in range(check.shape[0]):
    check_naive = check_naive +[word_feats(check["words"][i])]


# In[26]:


classifier = NaiveBayesClassifier.train(train_naive)
print("NLTK Naive bayes Accuracy : {}".format(nltk.classify.util.accuracy(classifier, test_naive)))
classifier.show_most_informative_features(5)


# In[27]:


y = []
only_words = [test_naive[i][0] for i in range(test.shape[0])]
for i in range(test.shape[0]):
    y = y + [classifier.classify(only_words[i])]
prediction["Naive"] = np.asarray(y)

y1 = []
for i in range(check.shape[0]):
    y1 = y1 + [classifier.classify(check_naive[i])]
    
check["Naive"] = y1


# In[28]:


from wordcloud import STOPWORDS

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
stopwords = set(STOPWORDS)
stopwords.remove("not")

count_vect = CountVectorizer(min_df = 2 , stop_words = stopwords, ngram_range=(1,2))
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(train["Summary_Clean"])
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_new_counts = count_vect.transform(test["Summary_Clean"])
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

checkcounts = count_vect.transform(check["Summary_Clean"])
checktfidf = tfidf_transformer.transform(checkcounts)


# In[30]:


from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB().fit(X_train_tfidf , train["senti"])
prediction['Multinomial'] = model1.predict_proba(X_test_tfidf)[:,1]
print("Multinomial Accuracy : {}".format(model1.score(X_test_tfidf , test["senti"])))

check["multi"] = model1.predict(checktfidf)


# In[31]:


from sklearn.naive_bayes import BernoulliNB
model2 = BernoulliNB().fit(X_train_tfidf,train["senti"])
prediction['Bernoulli'] = model2.predict_proba(X_test_tfidf)[:,1]
print("Bernoulli Accuracy : {}".format(model2.score(X_test_tfidf , test["senti"])))

check["Bill"] = model2.predict(checktfidf)


# In[33]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(solver='lbfgs' , C=1000)
logistic = logreg.fit(X_train_tfidf, train["senti"])
prediction['LogisticRegression'] = logreg.predict_proba(X_test_tfidf)[:,1]
print("Logistic Regression Accuracy : {}".format(logreg.score(X_test_tfidf , test["senti"])))

check["log"] = logreg.predict(checktfidf)


# In[35]:


words = count_vect.get_feature_names()
feature_coefs = pd.DataFrame(data = list(zip(words, logistic.coef_[0])), columns = ['feature', 'coef'])
feature_coefs.sort_values(by="coef")


# In[36]:


test.senti = test.senti.replace(["pos", "neg"], [True , False])


# In[38]:


from sklearn import metrics
keys = prediction.keys()
for key in ['Multinomial', 'Bernoulli', 'LogisticRegression']:
    print(" {}:".format(key))
    print(metrics.classification_report(test["senti"], prediction.get(key)>.5, target_names = ["positive", "negative"]))
    print("\n")


# In[39]:


def test_sample(model, sample):
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    prob = model.predict_proba(sample_tfidf)[0]
    if result.upper()=="POS":
        return 1;
    else:
        return 0;
    #print("sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))


# In[40]:


#test_sample(logreg, "The product was good and easy to use")


# In[41]:


#test_sample(logreg, "i hope you are fine")


# In[42]:


#test_sample(logreg, "you are stupid")


# In[43]:


#test_sample(logreg, "product is not good")


# In[44]:


#test_sample(logreg, "the whole experience was horrible and product is worst")


# In[ ]:





# In[ ]:




