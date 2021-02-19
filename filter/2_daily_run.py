import filtered_tweets as ft
import sendforlabel_tweets as sflt
import get_timezone as gt
from pathlib import Path
import os
import json
import pandas as pd

PATH = Path('/Users/peaceforlives/Documents/Projects/covid_tweets/data/')
DATA = Path('/Users/peaceforlives/Documents/Projects/cyberbullying/data/')
primary = PATH.joinpath(PATH.parents[0], 'filter/primary_keywords.txt')
secondary = PATH.joinpath(PATH.parents[0], 'filter/corona_keywords.txt')


# -------------------------------------------- #

# apply filtered_tweets function to all files in original folder
# it removes primary, secondary and additional keywords from tweets
# dump results in filtered folder

input_dir = PATH.joinpath(DATA, 'original')
output_dir = PATH.joinpath(PATH, 'filtered')

files_to_process = list( set( os.listdir(input_dir) ) - set( os.listdir(output_dir) ) ) # files already process should not be processed
files_to_process = [ i for i in files_to_process if '2020' in i]
files_to_process.sort()
# files_to_process = os.listdir(input_dir)
# files_to_process = ['tweets_2020-02-09.json']

for file in files_to_process[40:]:
    path_filename = PATH.joinpath(input_dir, file)       # join filename to input directory
    print(path_filename)                                 # print full path of the file being processed
    if '.DS_Store' in file:
        pass
    else:
        # apply filtered_tweets function to the file
        ft.filtered_tweets(path_infile=path_filename, primary=primary, secondary=secondary, additional=None, output_dir=output_dir)

# # get average tweet per day
# rows = []
# for file in os.listdir(output_dir):
#     path_filename = PATH.joinpath(output_dir, file)       # join filename to input directory
#     print(path_filename)                                  # print full path of the file being processed
#     if '.DS_Store' in file:
#         pass
#     else:
#         data = pd.read_json(path_filename, lines=True)
#     rows.append(data.shape[0])
# print( sum(rows)/len(rows) )

# # for days with corrupt json lines
# with open(path_filename, 'r') as datafile:
#     data = json.load(datafile)
# retail = pd.DataFrame(data)

# -------------------------------------------- #

# apply timezone function to all files in filtered folder
# it adds timezone, local time and state variables for each tweet
# dump results in located_tweets folder

input_dir = PATH.joinpath(PATH, 'filtered')       
output_dir = PATH.joinpath(PATH, 'located_tweets')
files_to_process = os.listdir(input_dir)
files_to_process.sort()
# file = 'tweets_2019-10-01.json'

for file in files_to_process: 
    path_filename = PATH.joinpath(input_dir, file)       # join filename to input directory
    print(path_filename)                                 # print full path of the file being processed
    if '.DS_Store' in file:
        pass
    else:
        # apply timezone function to file
        gt.timezone(path_infile=path_filename, output_dir=output_dir) 


# -------------------------------------------- #


# apply sendforlabel_tweets function to all files filtered located_tweets folder
# it count urls, hashtags and removes tweets with more than 5 hashtags, adds columns for labelling
# dump results in send_for_label folder
# DUE TO REMOVAL OF HTTPS, SOME TWEETS MAY NOT APPEAR IN SEND_FOR_LABEL FOLDER

input_dir = PATH.joinpath(PATH, 'located_tweets')
output_dir = PATH.joinpath(PATH, 'send_for_label')
files_to_process = os.listdir(input_dir)
files_to_process.sort()
# file = 'tweets_2020-02-09.json'
    
for file in files_to_process:
    path_filename = PATH.joinpath(input_dir, file)       # join filename to input directory
    print(path_filename)                                 # print full path of the file being processed
    if '.DS_Store' in file:
        pass
    else:
        # apply sendforlabel_tweets function to the file
        sflt.sendforlabel_tweets(path_infile=path_filename, output_dir=output_dir) 