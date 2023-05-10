from config import *
from datetime import datetime
from datetime import datetime
from collections import defaultdict


def tweet_count(id_str):
    return collection.count_documents({'user.id_str': id_str})


def tweeted_at_count(id_str):
    return collection.count_documents({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}})


def tweeted_at_count_month(chosen_airline_id):
    tweet_count_per_month = {}

    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': chosen_airline_id}
        }
    }):
        created_at = doc['created_at']
        created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
        if created_at_date.year == 2019 and 6 <= created_at_date.month <= 12:
            month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
            tweet_count_per_month.setdefault(month_year, 0)
            tweet_count_per_month[month_year] += 1
        elif created_at_date.year == 2020 and created_at_date.month <= 3:
            month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
            tweet_count_per_month.setdefault(month_year, 0)
            tweet_count_per_month[month_year] += 1

    return tweet_count_per_month


def replied_at_count_month(chosen_airline_id):
    reply_count_per_month = {}

    # Search for tweets where our airline was mentioned and retrieve their tweet IDs
    mention_docs = collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': chosen_airline_id}
        }
    }, {'id_str': 1})

    # Collect tweet IDs in a list
    mention_tweet_ids = [doc['id_str'] for doc in mention_docs]

    # Find tweets from our airline that are replies to the mentioned tweets
    reply_docs = collection.find({
        'user.id_str': chosen_airline_id,
        'in_reply_to_status_id_str': {'$in': mention_tweet_ids}
    })

    # Iterate over reply docs and count them per month
    for reply_doc in reply_docs:
        created_at = reply_doc['created_at']
        created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
        if (created_at_date.year == 2019 and 6 <= created_at_date.month <= 12) or (
                created_at_date.year == 2020 and created_at_date.month <= 3):
            month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
            reply_count_per_month.setdefault(month_year, 0)
            reply_count_per_month[month_year] += 1

    return reply_count_per_month


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
