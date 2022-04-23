#!/usr/bin/env python
# coding: utf-8

# In[212]:


# Importing dependencies
import json
import pandas as pd
import numpy as np
import os
import re
from sqlalchemy import create_engine
from config import db_password
import time


# In[107]:


# Setting base folder for resources
fileDir = 'Resources/'


# In[108]:


f'{fileDir}filename'


# In[110]:


# Opening 'wikipedia-movies.json', from the resources folder, in read mode.
with open(f'{fileDir}wikipedia-movies.json', mode='r') as file:
    # Loading json data to a variable
    wikiMoviesRaw = json.load(file)


# In[111]:


# Checking the amount of movies
len(wikiMoviesRaw)


# In[112]:


#Showing first 5 records
wikiMoviesRaw[:5]


# In[113]:


#Showing the last 5 records
wikiMoviesRaw[-5:]


# In[114]:


# Some records in the middle
wikiMoviesRaw[3600:3605]


# In[109]:


# Loading 'movies_metadata.csv' CSV file data to a variable
kaggleMetadata = pd.read_csv(f'{fileDir}movies_metadata.csv', low_memory=False)
# Loading 'ratings.csv' CSV file data to a variable
ratings = pd.read_csv(f'{fileDir}ratings.csv')


# In[115]:


# Inspecting data: 5 random rows of data.
kaggleMetadata.sample(n=5)


# In[116]:


# Turning the list of dictionaries (wikiMoviesRaw) into a DataFrame
wikiMoviesRawDf = pd.DataFrame(wikiMoviesRaw)
wikiMoviesRawDf.head()


# In[117]:


# Inspecting the column headers
wikiMoviesRawDf.columns.tolist()


# In[118]:


# Getting all movies that contain a 'Director' value and 'imdb_link'
wikiMovies = [movie for movie in wikiMoviesRaw
              if('Director' in movie or 'Directed by' in movie) and 'imdb_link' in movie]
len(wikiMovies)


# In[119]:


# Turning wikiMovies in to a DataFrame
wikiMoviesDf = pd.DataFrame(wikiMovies)
wikiMoviesDf.head()


# In[120]:


# Getting all movies that contain a 'Director' value, 'imdb_link', and NO 'No. of episodes'
wikiMovies = [movie for movie in wikiMoviesRaw
              if ('Director' in movie or 'Directed by' in movie)
              and 'imdb_link' in movie
              and 'No. of episodes' not in movie]

wikiMoviesDf = pd.DataFrame(wikiMovies)
wikiMoviesDf.head()


# In[122]:


square = lambda x: x * x
square(5)


# In[121]:


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


# In[123]:


#Checking what movies have a value for "Arabic"
wikiMoviesDf[wikiMoviesDf['Yiddish'].notnull()]['Yiddish']


# In[124]:


sorted(wikiMoviesDf.columns.tolist())


# In[125]:


clean_movies = [clean_movie(movie) for movie in wikiMovies]


# In[126]:


wikiMoviesDf = pd.DataFrame(clean_movies)
sorted(wikiMoviesDf.columns.tolist())


# In[127]:


wikiMoviesDf.head()


# In[128]:


wikiMoviesDf['imdb_id'] = wikiMoviesDf['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wikiMoviesDf))


# In[129]:


wikiMoviesDf.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wikiMoviesDf))


# In[130]:


[[column,wikiMoviesDf[column].isnull().sum()] for column in wikiMoviesDf.columns]


# In[131]:


wikiColumnsKeep = [column for column in wikiMoviesDf.columns if wikiMoviesDf[column].isnull().sum() < len(wikiMoviesDf) * 0.9]


# In[132]:


wikiMoviesDf = wikiMoviesDf[wikiColumnsKeep]
wikiMoviesDf


# In[133]:


boxOffice = wikiMoviesDf['Box office'].dropna()
def is_not_string(x):
    return type(x) != str


# In[134]:


boxOffice[boxOffice.map(is_not_string)]


# In[135]:


boxOffice[boxOffice.map(lambda x: type(x) !=str)]


# In[136]:


boxOffice = boxOffice.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[137]:


boxOffice.sample(20)


# In[138]:


formOne = r'\$\d+\.?\d*\s*[mb]illion'


# In[139]:


boxOffice.str.contains(formOne, flags=re.IGNORECASE, na=False).sum()


# In[140]:


formTwo = r'\$\d{1,3}(?:,\d{3})+'
boxOffice.str.contains(formTwo, flags=re.IGNORECASE, na=False).sum()


# In[141]:


matchesFormOne = boxOffice.str.contains(formOne, flags=re.IGNORECASE, na=False)
matchesFormTwo = boxOffice.str.contains(formTwo, flags=re.IGNORECASE, na=False)


# In[142]:


boxOffice[~matchesFormOne & ~matchesFormTwo]


# In[143]:


formOne = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
formTwo = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illi?on)'


# In[144]:


boxOffice = boxOffice.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[145]:


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


# In[146]:


wikiMoviesDf['box_office'] = boxOffice.str.extract(f'({formOne}|{formTwo})', flags=re.IGNORECASE)[0].apply(parse_dollars)
wikiMoviesDf['box_office']


# In[147]:


wikiMoviesDf


# In[148]:


wikiMoviesDf.drop('Box office', axis=1, inplace=True)


# In[149]:


wikiMoviesDf


# In[150]:


budget = wikiMoviesDf['Budget'].dropna()
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[151]:


matchesFromOne = budget.str.contains(formOne, flags=re.IGNORECASE, na=False)
matchesFormTwo = budget.str.contains(formTwo, flags=re.IGNORECASE, na= False)
budget[~matchesFromOne & ~matchesFormTwo]


# In[152]:


budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matchesFromOne & ~matchesFormTwo]


# In[153]:


wikiMoviesDf['budget'] = budget.str.extract(f'({formOne}|{formTwo})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[154]:


wikiMoviesDf.drop('Budget', axis=1, inplace=True)


# In[155]:


releaseDate = wikiMoviesDf['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[156]:


dateFromOne = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]?\d,\s\d{4}'
dateFormTwo = r'\d{4}.[01]\d.[0123]\d'
dateFormThree = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
dateFormFour = r'\d{4}'


# In[157]:


wikiMoviesDf['release_date'] = pd.to_datetime(releaseDate.str.extract(f'({dateFromOne}|{dateFormTwo}|{dateFormThree}|{dateFormFour})')[0], infer_datetime_format=True)


# In[158]:


runningTime = wikiMoviesDf['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[159]:


runningTime.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False).sum()


# In[160]:


runningTime[runningTime.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False) != True]


# In[161]:


runningTime.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False).sum()


# In[162]:


runningTime[runningTime.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False) != True]


# In[163]:


runningTimeExtract = runningTime.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[164]:


runningTimeExtract = runningTimeExtract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[165]:


wikiMoviesDf['running_time'] = runningTimeExtract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
wikiMoviesDf.drop('Running time', axis=1, inplace=True)


# In[166]:


wikiMoviesDf.head(10)


# In[167]:


kaggleMetadata.dtypes


# In[168]:


kaggleMetadata['adult'].value_counts()


# In[169]:


kaggleMetadata[~kaggleMetadata['adult'].isin(['True','False'])]


# In[170]:


kaggleMetadata = kaggleMetadata[kaggleMetadata['adult']=='False'].drop('adult',axis='columns')


# In[171]:


kaggleMetadata


# In[172]:


kaggleMetadata['video'].value_counts()


# In[173]:


kaggleMetadata['video'] = kaggleMetadata['video'] == 'True'


# In[174]:


kaggleMetadata


# In[175]:


kaggleMetadata['budget'] = kaggleMetadata['budget'].astype(int)
kaggleMetadata['id'] = pd.to_numeric(kaggleMetadata['id'], errors='raise')
kaggleMetadata['popularity'] = pd.to_numeric(kaggleMetadata['popularity'], errors='raise')


# In[176]:


kaggleMetadata['release_date'] = pd.to_datetime(kaggleMetadata['release_date'])


# In[177]:


kaggleMetadata.dtypes


# In[178]:


ratings.info(null_counts=True)


# In[179]:


pd.to_datetime(ratings['timestamp'], unit='s')


# In[180]:


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[181]:


pd.options.display.float_format = '{:20,.2f}'.format
ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[182]:


moviesDF = pd.merge(wikiMoviesDf, kaggleMetadata, on='imdb_id',suffixes=['_wiki','_kaggle'])


# In[183]:


# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle             Drop Wiki
# running_time             runtime                  Keep Kaggle; fill in zeros with wiki data
# budget_wiki              budget_kaggle            Keep Kaggle; fill in zeros with wiki data
# box_office               revenue                  Keep Kaggle; fill in zeros with wiki data
# release_date_wiki        release_date_kaggle      Drop Wiki
# Language                 original_language        Drop Wiki
# Production company(s)    production_companies     Drop Wiki


# In[184]:


moviesDF[['title_wiki', 'title_kaggle']]


# In[185]:


moviesDF[moviesDF['title_wiki'] != moviesDF['title_kaggle']][['title_wiki','title_kaggle']]


# In[186]:


# Show any rows where title_kaggle is empty
moviesDF[(moviesDF['title_kaggle']=='') | (moviesDF['title_kaggle'].isnull())]


# In[187]:


moviesDF.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[188]:


moviesDF.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[189]:


moviesDF.fillna(0).plot(x='box_office', y='revenue',kind='scatter')


# In[190]:


moviesDF.fillna(0)[moviesDF['box_office'] < 10**9].plot(x='box_office',y='revenue', kind='scatter')


# In[191]:


moviesDF[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[192]:


moviesDF[(moviesDF['release_date_wiki'] > '1996-01-01') & (moviesDF['release_date_kaggle'] < '1965-01-01')].index


# In[193]:


moviesDF = moviesDF.drop(moviesDF[(moviesDF['release_date_wiki'] > '1996-01-01') & (moviesDF['release_date_kaggle'] < '1965-01-01')].index)


# In[194]:


moviesDF[moviesDF['release_date_wiki'].isnull()]


# In[195]:


moviesDF['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[196]:


moviesDF['original_language'].value_counts(dropna=False)


# In[197]:


moviesDF[['Production company(s)', 'production_companies']]


# In[198]:


moviesDF.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[199]:


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column], axis=1
    )
    df.drop(columns=wiki_column, inplace=True)

moviesDF


# In[200]:


fill_missing_kaggle_data(moviesDF, 'runtime', 'running_time')
fill_missing_kaggle_data(moviesDF, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(moviesDF, 'revenue', 'box_office')
moviesDF


# In[201]:


for col in moviesDF.columns:
    listToTuples = lambda x: tuple(x) if type(x) == list else x
    valueCounts = moviesDF[col].apply(listToTuples).value_counts(dropna=False)
    numValues = len(valueCounts)
    if numValues == 1:
        print(col)


# In[202]:


moviesDF['video'].value_counts(dropna=False)


# In[203]:


moviesDF = moviesDF.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                              'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                              'genres','original_language','overview','spoken_languages','Country',
                              'production_companies','production_countries','Distributor',
                              'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                              ]]


# In[204]:


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
moviesDF


# In[205]:


ratingsCount = ratings.groupby(['movieId', 'rating'], as_index=False).count().rename({'userId':'count'}, axis=1).pivot(index='movieId',columns='rating',values='count')


# In[206]:


ratingsCount.columns = ['rating_' + str(col) for col in ratingsCount.columns]


# In[207]:


moviesWithRatingsDf = pd.merge(moviesDF, ratingsCount, left_on='kaggle_id', right_index=True, how='left')


# In[208]:


moviesWithRatingsDf[ratingsCount.columns] = moviesWithRatingsDf[ratingsCount.columns].fillna(0)


# In[209]:


dbString = f'postgresql://postgres:{db_password}@127.0.0.1:5432/movie_data'


# In[210]:


engine  = create_engine(dbString)


# In[211]:


moviesDF.to_sql(name='movies', con=engine)


# In[214]:


rowsImported = 0

startTime = time.time()
for data in pd.read_csv(f'{fileDir}ratings.csv', chunksize=1000000):

    print(f'importing rows {rowsImported} to {rowsImported + len(data)}...', end='')

    data.to_sql(name='ratings', con=engine, if_exists='append')
    rowsImported += len(data)

    print(f'Done. {time.time() - startTime} total seconds elapsed')


# In[ ]:




