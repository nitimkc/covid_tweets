
from pathlib import Path
import os
import json
import pandas as pd
import time 

from timezonefinder import TimezoneFinder
import datetime as dt
import pytz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


def get_timezone(tweet):
    
    locator = Nominatim(user_agent='myGeocoder', timeout=300)
    tf = TimezoneFinder() 

    tweet['timezone'], tweet['localtime'], tweet['state'] = [None]*3

    if tweet['coordinates']:
        # print(tweet)
        lat, lon = tweet['coordinates']['coordinates'][::-1]                      # get coordinates
        try:
            address = locator.reverse( str(lat)+' '+str(lon)).raw['address']      # get address from lookup
        except GeocoderTimedOut as e:
            print("timeout error")
            time.sleep(60)
            address = locator.reverse( str(lat)+' '+str(lon)).raw['address']      # get address from lookup

        if address['country_code'] in ['ca', 'us']:                               # select US or Canada only
            print(address['country'])
            tweet['timezone'] = tf.timezone_at(lng=lon, lat=lat)                  # convert to time format
            date = dt.datetime.strptime(tweet['created_at'], '%Y-%m-%d %H:%M:%S+00:00')
            date = date.replace(tzinfo = pytz.timezone('UTC'))
            tweet['localtime'] = date.astimezone(pytz.timezone(tweet['timezone'])).strftime('%Y-%m-%d %H:%M:%S %Z%z')    # get time of local zone
            tweet['state'] = address['state']                                                                       # get the state/province 
            # print(tweet['timezone'], tweet['localtime'], tweet['state'])

    return tweet

# tweet = [t for t in data if t['id']==11212257476130816000][0]
# file = 'tweets_2019-11-25.json'
# tweet = [t for t in data if t['coordinates']!=None][0]
# tweet
# get_timezone(tweet) 1212257476130816000

def timezone(path_infile, output_dir):
    
    data = []
    with open(path_infile) as f:
        for line in f:
            data.append(json.loads(line))
    
    path_outfile = Path.joinpath(output_dir, path_infile.name)
    for tweet in data:
        new_tweet = get_timezone(tweet)
        with open(path_outfile, 'a') as f:
            f.write(json.dumps(new_tweet) + "\n")