
# coding: utf-8

# ## Import

# In[1]:

import PyPDF2 #read the pdf

import matplotlib.pyplot as plt

import pandas as pdd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation


# ## Getting the data

# In[2]:

file = open('JavaBasics-notes.pdf', 'rb')
fileReader = PyPDF2.PdfFileReader(file)

total = fileReader.numPages


# #### Getting the function for feature name

# In[3]:

def get_topics(model, feature_names, no_top_words):
    all_ = []
    for topic_idx, topic in enumerate(model.components_):
        #print ("Topic %d:" % (topic_idx))
        x = " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        all_.append(str(x))
    return all_


# In[4]:

tra = []
for i in range(total):
  pg = fileReader.getPage(i)
  tra.append(pg.extractText())


# ### Algorithms:
#  NMF :Non-negative Matrix factorization      
#  LDA : Latent Derilicht Analysis

# In[5]:

documents = tra

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

idf = tfidf_vectorizer.idf_
x = dict(zip(tfidf_vectorizer.get_feature_names(), idf))

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = len(tra)

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)



# In[6]:

no_top_words = 10 #words for each page


# In[7]:

all_ = get_topics(nmf, tfidf_feature_names, no_top_words)#nmf


# In[8]:

all_2 = get_topics(lda, tf_feature_names, no_top_words)#lda


# ### Getting weights

# In[9]:

weights = {}
weights_2 = {}
for i in range(len(all_)):
  rest = all_[i].split(' ')
  rest2 = all_2[i].split(' ')
  for j in rest:
    if j in x:
      weights[str(j)] = x[str(j)]
  for k in rest2:
    if k in x:
      weights_2[str(k)] = x[str(k)]
        


# ### Making dataframe

# In[10]:

df1 = pdd.DataFrame(list(weights.items()), columns=['topic', 'weights'])


# In[11]:

df2 = pdd.DataFrame(list(weights_2.items()), columns=['topic', 'weights'])


# In[12]:

print(df1)


# In[13]:

print(df2)


# In[14]:

print('NMF')
for i in range(len(all_)):
    print('page = ', i, 'keywords : ' , all_[i])


# In[15]:

print('LDA')
for i in range(len(all_2)):
    print('page = ', i , 'keywords : ', all_2[i])


# In[ ]:



