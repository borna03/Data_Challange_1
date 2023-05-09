from config import *
from datetime import datetime


def tweet_count(id_str):
    return collection.count_documents({'user.id_str': id_str})


def tweeted_at_count(id_str):
    return collection.count_documents({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}})


def tweeted_at_lang(id_str):
    langs = {}
    for tweet in collection.find({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}}):
        if tweet['lang'] in langs:
            langs[tweet['lang']] += 1
        else:
            langs[tweet['lang']] = 1
    return langs


def responded_to_lang(id_str):
    langs = {}
    responses = [response for response in
                 collection.find({'user.id_str': id_str, 'in_reply_to_status_id_str': {'$ne': None}},
                                 {'in_reply_to_status_id_str': 1, "_id": 0}).distinct('in_reply_to_status_id_str')]

    for tweet in collection.find({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}},
                                  'in_reply_to_status_id_str': None}, {'id_str': 1, 'lang': 1, '_id': 0}):
        if tweet['id_str'] in responses:
            if tweet['lang'] in langs:
                langs[tweet['lang']] += 1
            else:
                langs[tweet['lang']] = 1
    return langs
