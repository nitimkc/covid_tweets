import glob 
from pathlib import Path
import pandas as pd
import os

# create random tweets to send for label
# make sure to not select a tweet already selected in previous random tweet file


PATH = r'/Users/peaceforlives/Documents/Projects/covid_tweets/data/send_for_label'
PATH_data = Path(PATH)                 # use your path

n = 1

# import previously selected random tweets 
in_files = glob.glob(os.path.join(PATH_data, "*.csv"))
insample = ( pd.read_csv(f) for f in in_files )

# combine all tweets from above to one object 
# select their id and set it as index

concat_insample = pd.concat(insample, ignore_index=True)
concat_insample['id'] = concat_insample['id'].astype(str)
concat_insample = concat_insample.set_index(concat_insample.id, inplace=False)
# concat_insample.head()
idx = concat_insample.index



# random_sample = nonsample_df.sample(n=8000)
random_sample = concat_insample.sample(n=8000)

# save every 1000 lines in new csv
n = range(1,9)
lines = range(1000, 9000, 1000)
for i,j in zip(n,lines):
    print(i,j)
    print(j-1000)
    path_outfile = Path.joinpath(PATH_data, 'random',  ('random_tweets_'+ str(i) + '.csv') )
    print(path_outfile)
    random_sample.iloc[(j-1000):j, ].to_csv(path_outfile, index=False, encoding = 'utf-8-sig')
