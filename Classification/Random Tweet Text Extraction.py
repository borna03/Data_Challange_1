from data_load import *
import pymongo

query = collection.aggregate([{'$sample': {'size': 10}}, {'$match': {'lang': 'en'}}])

# Iterate over all tweets, with live counter
for tweet in query:
    try:
        print(tweet['extended_tweet']['full_text'])
    except:
        print(tweet['text'])

