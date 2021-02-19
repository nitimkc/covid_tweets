from pathlib import Path
import pandas as pd
import json
import os

# path_infile = path_filename


def filter_keywords(data, keywords, omit=False):

    # load keywords
    filter = open(keywords, 'r', newline='')
    match = filter.read().splitlines()
    filter.close()

    # search for keywords in full text tweets
    data['full_tweet'] = data['full_tweet'].str.lower()
    to_match = '\\b(' + '|'.join(match) + ')\\b'
    matched_idx = data['full_tweet'].str.contains(to_match, case=False)

    # decision based on omit-whether the match is to be included or excluded 
    if omit==True:
        matched_data  = data[-matched_idx]
    else:
        matched_data = data[matched_idx]

    return matched_data

def filtered_tweets(path_infile, primary, secondary, additional, output_dir):
 
    # load and select reqd columns
    data = pd.read_json(path_infile, lines=True)
    # data = pd.DataFrame()
    # reader = pd.read_json(path_infile, lines=True, chunksize=1)
    # for i in reader:
    #     data.append(i)
    
    cols = ['created_at', 'id', 'text', 'source', 'geo', 'coordinates', 'place', 'lang', 'extended_tweet']
    data = data[cols]
    data['created_at'] = data['created_at'].astype(str)

    # extract full txt of extnd tweets
    data['full_tweet'] =  data['extended_tweet'].apply(pd.Series)['full_text']
    data['full_tweet'].fillna(data.text, inplace=True)
    data.drop(['text', 'extended_tweet'], axis=1, inplace=True)
    data.drop_duplicates(subset='id', keep='first', inplace=True)

    # load primary keywords
    primaryfilter_data = filter_keywords(data=data, keywords=primary)
    secondaryfilter_data = filter_keywords(data=primaryfilter_data, keywords=secondary)
    # additionalfilter_data = filter_keywords(data=secondaryfilter_data, keywords=additional, omit=True)

    # output
    path_outfile = Path.joinpath(output_dir, path_infile.name)
    secondaryfilter_data.to_json( path_outfile, orient='records', lines=True)
    # additionalfilter_data.to_json( path_outfile, orient='records', lines=True)


# filtered_tweets(path_infile, primary, secondary, additional, output_dir)


# from pathlib import Path
# import os
# import json
# import pandas as pd

# PATH = Path('/Users/peaceforlives/Documents/Projects/cyberbullying/data/')
# file = 'tweets_2020-02-09.json'
# input_dir = PATH.joinpath(PATH, 'original')
# path_filename = PATH.joinpath(input_dir, file)
# print(path_filename) 
# path_infile = path_filename

# with open(path_infile) as f:
#     contents = f.read()
# string_data = contents.split("\n")
# list_data = []
# removed_items = []
# for i in range(0,len(string_data)):
#     try:    
#         list_data.append(json.loads(string_data[i]))
#     except:
#         removed_items.append(i)
#         pass
# data = pd.DataFrame.from_records(list_data)


