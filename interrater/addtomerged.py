###########################################
# add more labelled tweets from annotators
# to file merged.json
###########################################

import pandas as pd
import json
import os
from pathlib import Path


ROOT = Path(r'/Users/peaceforlives/Documents/Projects/covid_tweets/')
LABELLED_XLSX = Path(os.path.join(ROOT, 'data', 'labelled_tweets'))
LABELLED_CORPUS = Path(os.path.join(ROOT, 'data', 'labelled_tweets', 'ab'))


###########################################
# load new labelled tweets from annotators
# and save as json
###########################################

 
files = [x for x in list(LABELLED_XLSX.iterdir()) if x.name.endswith('.xlsx')] # select files in dir
files.sort()                                                    
all = pd.DataFrame()
for f in files:
    print(f)
    labl = pd.read_excel(f, usecols=[0,1,2,3,4,5], index_col=None, encoding='utf-8-sig') # read excel file                        
    labl['bullying_role'].replace(['Assistant', 'Bystander', 'Bully', 'Reinforcer'], 'Other', inplace=True)
    labl['form_of_bullying'].replace(['Physical', 'Verbal'], 'General', inplace=True)
    print(labl.columns)
    labl.to_json(f.with_suffix('.json'), orient='records', lines=True)                   # save as json
    all = all.append(labl)                                                               # combine all files in one
    print(all.shape)
    all = all.apply(lambda x: x.astype(str).str.lower())
all['bullying_role'].value_counts()
all['form_of_bullying'].value_counts()


###########################################
# save as json
###########################################

# all.to_json(Path.joinpath(LABELLED_CORPUS, 'merged1.json'), orient='records', lines=True)

# for index, eachrow in all.iloc[:5,].iterrows():
#     print(dict(eachrow))
for index, eachrow in all.iterrows():
    with open(Path.joinpath(LABELLED_CORPUS,'merged.json'), 'a') as f:
        f.write("\n" + json.dumps(dict(eachrow)))

