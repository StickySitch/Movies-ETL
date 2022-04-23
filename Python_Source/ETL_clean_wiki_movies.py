#!/usr/bin/env python
# coding: utf-8

# In[84]:


import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import time
import psycopg2


# In[85]:


# 1. Add the clean movie function that takes in the argument, "movie".
def clean_movie(movie):
    #creating a non-destructive copy
    movie = dict(movie)
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese',            'Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)

    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles
    def change_column_name(old_name,new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)

    # Merging columns with the same values
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie


# In[85]:





# In[86]:


# 2 Add the function that takes in three arguments;
# Wikipedia data, Kaggle metadata, and MovieLens rating data (from Kaggle)

def extract_transform_load(wikiData, kaggleData, lensRatingData):
    # Read in the kaggle metadata and MovieLens ratings CSV files as Pandas DataFrames.
    kaggleMeta = pd.read_csv(kaggleData, low_memory=False)
    ratingLens = pd.read_csv(lensRatingData)

    # Open and read the Wikipedia data JSON file.
    with open (wikiData, mode='r') as file:
        wikiMoviesRaw = json.load(file)
    
    # 3. Write a list comprehension to filter out TV shows.
    wikiMovies = [movie for movie in wikiMoviesRaw
                  if('Director' in movie or 'Directed by' in movie) and 'imdb_link'
                  and 'No. of episodes' not in movie]

    # 4. Write a list comprehension to iterate through the cleaned wiki movies list
    # and call the clean_movie function on each movie.
    cleanWikiMovies = [clean_movie(movie) for movie in wikiMovies]

    # 5. Read in the cleaned movies list from Step 4 as a DataFrame.
    cleanWikiMoviesDf = pd.DataFrame(cleanWikiMovies)


    # 6. Write a try-except block to catch errors while extracting the IMDb ID using a regular expression string and
    #  dropping any imdb_id duplicates. If there is an error, capture and print the exception.
    try:
        cleanWikiMoviesDf['imdb_id'] = cleanWikiMoviesDf['imdb_link'].str.extract(r'(tt\d{7})')
        cleanWikiMoviesDf.drop_duplicates(subset='imdb_id', inplace=True)
    except ValueError as error:
        print(f'Sorry, {error}')


    #  7. Write a list comprehension to keep the columns that don't have null values from the wiki_movies_df DataFrame.
    wikiKeepColumns = [column for column in cleanWikiMoviesDf.columns if cleanWikiMoviesDf[column].isnull().sum() < len(cleanWikiMoviesDf)]
    wikiMoviesDf = cleanWikiMoviesDf[wikiKeepColumns]

    # 8. Create a variable that will hold the non-null values from the “Box office” column.
    # 9. Convert the box office data created in Step 8 to string values using the lambda and join functions.
    boxOffice = wikiMoviesDf['Box office'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # 10. Write a regular expression to match the six elements of "form_one" of the box office data.
    formOne = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    # 11. Write a regular expression to match the three elements of "form_two" of the box office data.
    formTwo = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illi?on)'


# 12. Add the parse_dollars function.
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan
        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]', '', s)
            # convert to float and multiply by a million
            value = float(s) * 10**6
            # return value
            return value
        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on',s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s= re.sub('\$|\s|[a-zA-Z]', '', s)
            # convert to float and multiply by a billion
            value = float(s) * 10**9
            # return value
            return value
        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)',s,flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub('\$|,','',s)
            # convert to float
            value = float(s)
            # return value
            return value
        # otherwise, return NaN
        else:
            return np.nan
    
        
    # 13. Clean the box office column in the wiki_movies_df DataFrame.
    boxOffice = boxOffice.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    wikiMoviesDf['box_office'] = boxOffice.str.extract(f'({formOne}|{formTwo})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wikiMoviesDf.drop('Box office', axis=1, inplace=True)


    # 14. Clean the budget column in the wiki_movies_df DataFrame.
    budget = wikiMoviesDf['Budget'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    wikiMoviesDf['budget'] = budget.str.extract(f'({formOne}|{formTwo})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wikiMoviesDf.drop('Budget', axis=1, inplace=True)

    # 15. Clean the release date column in the wiki_movies_df DataFrame.
    releaseDate = wikiMoviesDf['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    dateFromOne = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]?\d,\s\d{4}'
    dateFormTwo = r'\d{4}.[01]\d.[0123]\d'
    dateFormThree = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    dateFormFour = r'\d{4}'
    wikiMoviesDf['release_date'] = pd.to_datetime(releaseDate.str.extract(f'({dateFromOne}|{dateFormTwo}|{dateFormThree}|{dateFormFour})')[0], infer_datetime_format=True)
    wikiMoviesDf.drop('Release date', axis=1, inplace=True)


    # 16. Clean the running time column in the wiki_movies_df DataFrame.
    runningTime = wikiMoviesDf['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    runningTimeExtract = runningTime.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

    runningTimeExtract = runningTimeExtract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    wikiMoviesDf['running_time'] = runningTimeExtract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    wikiMoviesDf.drop('Running time', axis=1, inplace=True)

    # Return three variables. The first is the wiki_movies_df DataFrame
    
    return wikiMoviesDf, kaggleMeta, ratingLens


# In[87]:


# 17. Create the path to your file directory and variables for the three files.
file_dir = 'Resources'
# The Wikipedia data
wiki_file = f'{file_dir}/wikipedia-movies.json'
# The Kaggle metadata
kaggle_file = f'{file_dir}/movies_metadata.csv'
# The MovieLens rating data.
ratings_file = f'{file_dir}/ratings.csv'


# In[88]:


# 18. Set the three variables equal to the function created in D1.
wiki_file, kaggle_file, ratings_file = extract_transform_load(wiki_file,kaggle_file,ratings_file)


# In[89]:


# 19. Set the wiki_movies_df equal to the wiki_file variable. 
wiki_movies_df = wiki_file


# In[90]:


# 20. Check that the wiki_movies_df DataFrame looks like this.
wiki_movies_df.drop(wiki_movies_df.loc[:, 'Genre':'Color process'].columns, axis=1, inplace=True)
wiki_movies_df.head(20)


# In[91]:


# 21. Check that wiki_movies_df DataFrame columns are correct. 
wiki_movies_df.columns.to_list()


# In[91]:




