###########################################
# calculate kappa
# sample tweets from each annotator 
###########################################

import pandas as pd
import json
import os
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

ROOT = Path(r'/Users/peaceforlives/Documents/Projects/covid_tweets/')
INTERRATER = Path(os.path.join(ROOT, 'interrater'))
LABELLED_A_CORPUS = Path(os.path.join(ROOT, 'data', 'labelled_tweets', 'a'))
LABELLED_B_CORPUS = Path(os.path.join(ROOT, 'data', 'labelled_tweets', 'b'))
LABELLED_CORPUS = Path(os.path.join(ROOT, 'data', 'labelled_tweets', 'ab'))


###########################################
# merge tweets from each annotator
# save each file as json
###########################################

full_data = []    
for i in [LABELLED_A_CORPUS, LABELLED_B_CORPUS]:
    labelled_files = list(i.iterdir())
    files = [x for x in labelled_files if x.name.endswith('.xlsx')] # select files in dir
    files.sort()                                                    # sort
    all = pd.DataFrame()
    for f in files:
        print(f)
        labl = pd.read_excel(f, usecols=[0,1,2,3,4,5], index_col=None, encoding='utf-8-sig') # read excel file                        
        labl['bullying_role'].replace(['Assistant', 'Bystander', 'Bully', 'Reinforcer'], 'Other', inplace=True)
        labl['form_of_bullying'].replace(['Physical', 'Verbal'], 'General', inplace=True)
        labl.to_json(f.with_suffix('.json'), orient='records', lines=True)                   # save as json
        all = all.append(labl)                                                               # combine all files in one
        print(all.shape)
    all = all.apply(lambda x: x.astype(str).str.lower())
    full_data.append( all ) 

full_a = full_data[0]
full_b = full_data[1]
for i in labl.columns[2:]:
    print(labl[i].unique())
    print(labl[i].value_counts())

###########################################
# sample 1500 tweets from each annotator 
# save as json
###########################################

id_a = full_a['id'].sample(1500)
id_b = list(set(full_b['id']) - set(id_a))
combined = pd.concat([ full_a[full_a['id'].isin(id_a)], full_b[full_b['id'].isin(id_b)] ])
combined.to_json(Path.joinpath(LABELLED_CORPUS, 'merged.json'), orient='records', lines=True)
for i in combined.columns[2:]:
    print(combined[i].value_counts())


###########################################
# combine annotation from both annotators
###########################################

merged_df = full_a.copy()
merged_df.set_index('id', inplace=True)
merged_df['bullying_trace_b'] = full_b['bullying_trace'].values
merged_df['bullying_role_b'] = full_b['bullying_role'].values
merged_df['form_of_bullying_b'] = full_b['form_of_bullying'].values
merged_df['bullying_post_type_b'] = full_b['bullying_post_type'].values
merged_df['full_tweet'] = [i.encode('utf-16','replace').decode('utf-16') for i in merged_df.full_tweet]
# merged_df.to_excel(r'merged_df.xlsx', index = False, encoding = 'utf16')


###########################################
# calculate kappa score
###########################################

label = 'bullying_trace'
cohen_kappa_score(merged_df[label], merged_df[label+'_b'], labels=['yes', 'no'])

# from nltk import agreement
# import numpy as np
# rater1 = np.array(merged_df[label])
# rater2 = np.array(merged_df[label+'_b'])
# taskdata=[ [0,str(i),str(rater1[i])] for i in range(0,len(rater1))]+[[1,str(i),str(rater2[i])] for i in range(0,len(rater2))]
# ratingtask = agreement.AnnotationTask(data=taskdata)
# print("kappa " + str(ratingtask.kappa()))

test = merged_df[[label, label+'_b']].copy()
test = test.apply(lambda x: x.str.strip())
test.dropna(axis=1, inplace=True)

cohen_kappa_score(test[label], test[label+'_b'])
matthews_corrcoef(test[label], test[label+'_b'], sample_weight=None)
cf = confusion_matrix(test[label], test[label+'_b'])
pd.DataFrame(cf).to_clipboard()

#---------------------------------------------------------------------------

label = 'bullying_role'
# cohen_kappa_score(merged_df[label], merged_df[label+'_b'])

test = merged_df[[label, label+'_b']].copy()
test = test.apply( lambda x: x.str.strip() )
vals_to_replace = {'bystander':'other', 'reinforcer':'other', 'assistant':'other'}
test[label] = test[label].replace(vals_to_replace)
test[label+'_b'] = test[label+'_b'].replace(vals_to_replace)
cohen_kappa_score(test[label], test[label+'_b'])
matthews_corrcoef(test[label], test[label+'_b'], sample_weight=None)

lab = ['nan', 'reporter', 'defender', 'accuser', 'victim', 'bully', 'other']
cf = confusion_matrix(test[label], test[label+'_b'], labels=lab)
pd.DataFrame(cf).to_clipboard()

# without nan
# test = test[~test.stack().str.contains('remove|nan').any(level=0)]
idx = (test[label]=='nan')& (test[label+'_b']=='nan')
test = test.loc[~idx,:]
cohen_kappa_score(test[label], test[label+'_b'])
matthews_corrcoef(test[label], test[label+'_b'], sample_weight=None)

cf = confusion_matrix(test[label], test[label+'_b'], labels=lab)
pd.DataFrame(cf).to_clipboard()

#---------------------------------------------------------------------------

label = 'form_of_bullying'
# cohen_kappa_score(merged_df[label], merged_df[label+'_b'])

test = merged_df[[label+'_b', label]].copy()
test = test.apply(lambda x: x.str.strip())
lab = ['nan', 'general', 'xenophobia', 'cyber', 'verbal', 'physical']
cf = confusion_matrix(test[label], test[label+'_b'], labels=lab)
pd.DataFrame(cf).to_clipboard()

# without nan
# test = test[~test.stack().str.contains('remove|nan').any(level=0)]
idx = (test[label]=='nan')& (test[label+'_b']=='nan')
test = test.loc[~idx,:]
cohen_kappa_score(test[label], test[label+'_b'])
cf = confusion_matrix(test[label], test[label+'_b'], labels=lab)
pd.DataFrame(cf).to_clipboard()

#---------------------------------------------------------------------------

label = 'bullying_post_type'
# cohen_kappa_score(merged_df[label], merged_df[label+'_b'])

test = merged_df[[label+'_b', label]].copy()
test = test.apply(lambda x: x.str.strip())
lab = ['nan', 'reports', 'accusations', 'self-disclosures', 'cyberbullying', 'denials']
cf = confusion_matrix(test[label], test[label+'_b'], labels=lab)
pd.DataFrame(cf).to_clipboard()

# without nan
# test = test[~test.stack().str.contains('remove|nan').any(level=0)]
idx = (test[label]=='nan')& (test[label+'_b']=='nan')
test = test.loc[~idx,:]
cohen_kappa_score(test[label], test[label+'_b'])
cf = confusion_matrix(test[label], test[label+'_b'], labels=lab)
pd.DataFrame(cf).to_clipboard()
