#!/usr/bin/env python
# coding: utf-8

# In[56]:


import json
import pandas as pd
import numpy as np

import re

from sqlalchemy import create_engine
import psycopg2

from config import db_password

import time


# In[57]:


#  Add the clean movie function that takes in the argument, "movie".
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


# In[58]:


# 1 Add the function that takes in three arguments;
# Wikipedia data, Kaggle metadata, and MovieLens rating data (from Kaggle)

def extract_transform_load(wikiData, kaggleData, lensRatingData):
    # Read in the kaggle metadata and MovieLens ratings CSV files as Pandas DataFrames.
    kaggleMetadata = pd.read_csv(kaggleData, low_memory=False)
    ratingLens = pd.read_csv(lensRatingData)

    # Open and read the Wikipedia data JSON file.
    with open (wikiData, mode='r') as file:
        wikiMoviesRaw = json.load(file)
    
    # Write a list comprehension to filter out TV shows.
    wikiMovies = [movie for movie in wikiMoviesRaw
                  if('Director' in movie or 'Directed by' in movie) and 'imdb_link'
                  and 'No. of episodes' not in movie]

    # Write a list comprehension to iterate through the cleaned wiki movies list
    # and call the clean_movie function on each movie.
    cleanWikiMovies = [clean_movie(movie) for movie in wikiMovies]

    # Read in the cleaned movies list from Step 4 as a DataFrame.
    wikiMoviesDf = pd.DataFrame(cleanWikiMovies)

    # Write a try-except block to catch errors while extracting the IMDb ID using a regular expression string and
    #  dropping any imdb_id duplicates. If there is an error, capture and print the exception.
    try:
        wikiMoviesDf['imdb_id'] = wikiMoviesDf['imdb_link'].str.extract(r'(tt\d{7})')
        wikiMoviesDf.drop_duplicates(subset='imdb_id', inplace=True)
    except ValueError as error:
        print(f'Sorry, {error}')
    #  Write a list comprehension to keep the columns that don't have null values from the wiki_movies_df DataFrame.
    wikiKeepColumns = [column for column in wikiMoviesDf.columns if wikiMoviesDf[column].isnull().sum() < len(wikiMoviesDf)]
    wikiMoviesDf = wikiMoviesDf[wikiKeepColumns]
    

    # Create a variable that will hold the non-null values from the “Box office” column.
    # Convert the box office data created in Step 8 to string values using the lambda and join functions.
    boxOffice = wikiMoviesDf['Box office'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    

    

    # Write a regular expression to match the six elements of "form_one" of the box office data.
    formOne = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    # Write a regular expression to match the three elements of "form_two" of the box office data.
    formTwo = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illi?on)'

    # Add the parse_dollars function.
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
        
    # Clean the box office column in the wiki_movies_df DataFrame.
    boxOffice = boxOffice.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    wikiMoviesDf['box_office'] = boxOffice.str.extract(f'({formOne}|{formTwo})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wikiMoviesDf.drop('Box office', axis=1, inplace=True)


    # Clean the budget column in the wiki_movies_df DataFrame.
    budget = wikiMoviesDf['Budget'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    wikiMoviesDf['budget'] = budget.str.extract(f'({formOne}|{formTwo})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wikiMoviesDf.drop('Budget', axis=1, inplace=True)

    # Clean the release date column in the wiki_movies_df DataFrame.
    releaseDate = wikiMoviesDf['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    dateFromOne = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]?\d,\s\d{4}'
    dateFormTwo = r'\d{4}.[01]\d.[0123]\d'
    dateFormThree = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    dateFormFour = r'\d{4}'
    wikiMoviesDf['release_date'] = pd.to_datetime(releaseDate.str.extract(f'({dateFromOne}|{dateFormTwo}|{dateFormThree}|{dateFormFour})')[0], infer_datetime_format=True)
    wikiMoviesDf.drop('Release date', axis=1, inplace=True)

    # Clean the running time column in the wiki_movies_df DataFrame.
    runningTime = wikiMoviesDf['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    runningTimeExtract = runningTime.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

    runningTimeExtract = runningTimeExtract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    wikiMoviesDf['running_time'] = runningTimeExtract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    wikiMoviesDf.drop('Running time', axis=1, inplace=True)
     
    # 2. Clean the Kaggle metadata.
    kaggleMetadata = kaggleMetadata[kaggleMetadata['adult']=='False'].drop('adult',axis='columns')
    kaggleMetadata['video'] = kaggleMetadata['video'] == 'True'
    kaggleMetadata['budget'] = kaggleMetadata['budget'].astype(int)
    kaggleMetadata['id'] = pd.to_numeric(kaggleMetadata['id'], errors='raise')
    kaggleMetadata['popularity'] = pd.to_numeric(kaggleMetadata['popularity'], errors='raise')
    kaggleMetadata['release_date'] = pd.to_datetime(kaggleMetadata['release_date'])


    # 3. Merged the two DataFrames into the movies DataFrame.
    moviesDF = pd.merge(wikiMoviesDf, kaggleMetadata, on='imdb_id',suffixes=['_wiki','_kaggle'])

    # 4. Drop unnecessary columns from the merged DataFrame.
    moviesDF = moviesDF.drop(moviesDF[(moviesDF['release_date_wiki'] > '1996-01-01') & (moviesDF['release_date_kaggle'] < '1965-01-01')].index)
    moviesDF.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
    # 5. Add in the function to fill in the missing Kaggle data.
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column], axis=1
        )
        df.drop(columns=wiki_column, inplace=True)

    # 6. Call the function in Step 5 with the DataFrame and columns as the arguments.
    fill_missing_kaggle_data(moviesDF, 'runtime', 'running_time')
    fill_missing_kaggle_data(moviesDF, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(moviesDF, 'revenue', 'box_office')


    # 7. Filter the movies DataFrame for specific columns.
    for col in moviesDF.columns:
        listToTuples = lambda x: tuple(x) if type(x) == list else x
    valueCounts = moviesDF[col].apply(listToTuples).value_counts(dropna=False)
    numValues = len(valueCounts)
    if numValues == 1:
        print(col)

    moviesDF = moviesDF.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                                'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                                'genres','original_language','overview','spoken_languages','Country',
                                'production_companies','production_countries','Distributor',
                                'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                                ]]

    # 8. Rename the columns in the movies DataFrame.
    moviesDF.rename({'id':'kaggle_id',
                     'title_kaggle':'title',
                     'url':'wikipedia_url',
                     'budget_kaggle':'budget',
                     'release_date_kaggle':'release_date',
                     'Country':'country',
                     'Distributor':'distributor',
                     'Producer(s)':'producers',
                     'Director':'director',
                     'Starring':'starring',
                     'Cinematography':'cinematography',
                     'Editor(s)':'editors',
                     'Writer(s)':'writers',
                     'Composer(s)':'composers',
                     'Based on':'based_on'
                     }, axis='columns', inplace=True)

    # 9. Transform and merge the ratings DataFrame.
    ratingsCount = ratingLens.groupby(['movieId', 'rating'], as_index=False).count().rename({'userId':'count'}, axis=1).pivot(index='movieId',columns='rating',values='count')
    ratingsCount.columns = ['rating_' + str(col) for col in ratingsCount.columns]
    moviesWithRatingsDf = pd.merge(moviesDF, ratingsCount, left_on='kaggle_id', right_index=True, how='left')
    moviesWithRatingsDf[ratingsCount.columns] = moviesWithRatingsDf[ratingsCount.columns].fillna(0)

    #Creating connection to DataBase
    dbString = f'postgresql://postgres:{db_password}@127.0.0.1:5432/movie_data'
    engine = create_engine(dbString)

    #Adding to the database
    with engine.connect() as conn, conn.begin():
        moviesDF.to_sql(name='movies', con=engine, if_exists='replace')

    #Importing ratings DF to database in chunks
    with engine.connect() as conn, conn.begin():
        rowsImported = 0
        startTime = time.time()
        for data in pd.read_csv(lensRatingData, chunksize=1000000):
            print(f'importing rows {rowsImported} to {rowsImported + len(data)}...', end='')

            data.to_sql(name='ratings', con=engine, if_exists='append')
            rowsImported += len(data)

            print(f'Done. {time.time() - startTime} total seconds elapsed')

    return wikiMoviesDf, moviesWithRatingsDf, moviesDF


# In[59]:


# 10. Create the path to your file directory and variables for the three files.
file_dir = 'Resources'
# The Wikipedia data
wiki_file = f'{file_dir}/wikipedia-movies.json'
# The Kaggle metadata
kaggle_file = f'{file_dir}/movies_metadata.csv'
# The MovieLens rating data.
ratings_file = f'{file_dir}/ratings.csv'


# In[60]:


# 11. Set the three variables equal to the function created in D1.
wiki_file, kaggle_file, ratings_file = extract_transform_load(wiki_file, kaggle_file, ratings_file)


# In[61]:


# 12. Set the DataFrames from the return statement equal to the file names in Step 11. 
wiki_movies_df = wiki_file
movies_with_ratings_df = kaggle_file
movies_df = ratings_file


# In[62]:


# 13. Check the wiki_movies_df DataFrame.
wiki_movies_df.drop(wiki_movies_df.loc[:, 'Genre':'Color process'].columns, axis=1, inplace=True)


# In[63]:


wiki_file.head()


# In[64]:


# 14. Check the movies_with_ratings_df DataFrame.
movies_with_ratings_df.head()


# In[65]:


# 15. Check the movies_df DataFrame. 
movies_df.head()


# In[65]:




