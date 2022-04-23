#!/usr/bin/env python
# coding: utf-8

# In[6]:


import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import psycopg2
from config import db_password
import time

# 1. Create a function that takes in three arguments;
# Wikipedia data, Kaggle metadata, and MovieLens rating data (from Kaggle)

def extract_transform_load(wikiData, kaggleData, lensRatingData):
    # 2. Read in the kaggle metadata and MovieLens ratings CSV files as Pandas DataFrames.
    kaggleMeta = pd.read_csv(kaggleData, low_memory=False)
    ratingLens = pd.read_csv(lensRatingData)
    

    # 3. Open the read the Wikipedia data JSON file.
    with open (wikiData, mode='r') as file:
        wikiMoviesRaw = json.load(file)
    # 4. Read in the raw wiki movie data as a Pandas DataFrame.
    wikiDataDf = pd.DataFrame(wikiMoviesRaw)
    
    # 5. Return the three DataFrames
    return wikiDataDf, kaggleMeta, ratingLens

# 6 Create the path to your file directory and variables for the three files. 
file_dir = 'Resources/'
# Wikipedia data
wiki_file = f'{file_dir}wikipedia-movies.json'
# Kaggle metadata
kaggle_file = f'{file_dir}movies_metadata.csv'
# MovieLens rating data.
ratings_file = f'{file_dir}ratings.csv'

# 7. Set the three variables in Step 6 equal to the function created in Step 1.
wiki_file, kaggle_file, ratings_file = extract_transform_load(wiki_file, kaggle_file, ratings_file)


# In[7]:


# 8. Set the DataFrames from the return statement equal to the file names in Step 6. 
wiki_movies_df = wiki_file
kaggle_metadata = kaggle_file
ratings = ratings_file


# In[ ]:





# In[8]:


# 9. Check the wiki_movies_df DataFrame.
wiki_movies_df.head()


# In[9]:


# 10. Check the kaggle_metadata DataFrame.
kaggle_metadata.head()


# In[10]:


# 11. Check the ratings DataFrame.
ratings.sample(20)


# In[5]:




