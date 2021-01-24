#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim.downloader as api
import gensim


# In[2]:


from gensim.models import KeyedVectors


# In[3]:


model = gensim.downloader.load("glove-twitter-50")


# In[4]:


model.save('glove-twitter-50')


# In[ ]:


word2vec = gensim.models.keyedvectors.KeyedVectors.load("glove-twitter-50")


# In[ ]:


download.file("http://nlp.stanford.edu/data/glove.twitter.27B.zip", destfile = "glove.zip")


# In[5]:


#Test
#word2vec = gensim.models.keyedvectors.KeyedVectors.load("glove-twitter-50")


# In[ ]:




