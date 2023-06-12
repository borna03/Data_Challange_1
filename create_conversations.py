from pymongo import UpdateOne, UpdateMany
from data_load import *
import pandas as pd


# Function to (re)set a column
def set_new_column(column_name, column_value):
    update_operation = {"$set": {column_name: column_value}}
    collection.update_many({}, update_operation)
    print('done')


# Add the replies to each tweet to the database
def add_replies():
    tweets = collection.find()
    bulk_updates = []

    # Iterate over your documents and define the update operation for each document
    count = 0
    for tweet in tweets:
        if tweet['in_reply_to_status_id_str'] is not None:
            filter = {"id_str": tweet["in_reply_to_status_id_str"]}  # Modify the filter based on your criteria
            update_operation = {"$push": {"replies": {"$each": [tweet['id_str']]}}}
            update = UpdateOne(filter, update_operation)
            bulk_updates.append(update)
        count += 1
        print(count)

    # Perform the bulk write operation
    print('doing the bulk update now')
    collection.bulk_write(bulk_updates)
    print('done')


# Function to create the sub-conversations
def subdivide_conversations():
    # Gather all conversations grouped by conversation_id
    pipeline = [{
        "$group": {
            "_id": "$conversation_id",
            "tweets": {"$push": "$$ROOT"}
        }}]
    conversations = collection.aggregate(pipeline)

    # Initialise bulk_updates
    bulk_updates = []

    # Loop over all general conversations
    for convo in conversations:
        # Get the levels and chain of every leaf tweet in the conversation
        tweets = {tweet['id_str']: tweet['in_reply_to_status_id_str'] for tweet in convo}
        tweet_info = determine_levels(convo, tweets)
        subconversation_id = 1

        # Loop over all tweets in the conversation
        for tweet in convo:

            # Check if it is a leaf node and if it isn't handled
            if not tweet['replies'] and tweet_info[tweet['id_str']]['handled'] is False:
                subconvo_tweets = []
                # Add all tweets on the same level that reply to the same tweet to the same conversation
                for key, value in tweet_info:
                    if value['level'] == tweet_info[tweet['id_str']]['level'] and \
                            tweets[tweet['id_str']] == tweets[tweet[key]]:
                        subconvo_tweets.append(key)
                        value['handled'] = True

                subconvo_tweets.extend(tweets[tweet['id_str']])
                filter = {"id_str": {'$in': subconvo_tweets}}
                update_operation = {"$set": {"sub-conversation_id": subconversation_id}}
                update = UpdateOne(filter, update_operation)
                bulk_updates.append(update)
                subconversation_id += 1

    collection.bulk_write(bulk_updates)


# Determine the levels of the tweets in a given conversation
def determine_levels(conversation, tweet_chain):
    levels = {}
    for tweet in conversation:
        if not tweet['replies']:
            levels[tweet['id_str']]['level'] = levels_count(tweet_chain, tweet["id_str"], 0)
            levels[tweet['id_str']]['chain'] = get_tweet_chain(tweet_chain, tweet['id_str'])
            levels[tweet['id_str']]['handled'] = False
    return levels


# Get all tweets in a chain (given a leaf node)
def get_tweet_chain(tweet_chain, id_str):
    if id_str is None:
        return []
    else:
        chain = [id_str]
        if tweet_chain.get(id_str):
            chain.extend(get_tweet_chain(tweet_chain, tweet_chain.get(id_str)))
        return chain


# Function that determines the level of all tweets in a conversation
def levels_count(tweet_chain, id_str, level):
    if tweet_chain[id_str] is None:
        return level + 1
    else:
        return levels_count(tweet_chain, tweet_chain[id_str], level) + 1


# Add the general conversation id
def general_conversation():
    # Get all root tweets
    root_tweets = collection.find({'in_reply_to_status_id_str': None})

    # Initialise conversation_id, and bulk_updates
    conversation_id = 1
    bulk_updates = []

    # Loop over all root tweets
    for tweet in root_tweets:
        # Query all replies in the chain
        conversation = collection.aggregate([
            {"$match": {"id_str": tweet['id_str']}},
            {
                "$graphLookup": {
                    "from": "conversations",
                    "startWith": "$id_str",
                    "connectFromField": "id_str",
                    "connectToField": "replies",
                    "as": "reply_chain"
                }
            }
        ])

        # Convert to a list of id's
        id_list = [tweet['id_str'] for tweet in conversation]

        # Add an update operation to bulk_updates
        filter = {"id_str": {"$in": id_list}}
        update_operation = {"$set": {"conversation_id": conversation_id}}
        update = UpdateMany(filter, update_operation)
        bulk_updates.append(update)

        # Up the conversation_id
        conversation_id += 1

    # Perform all updates in bulk_updates
    collection.bulk_write(bulk_updates)
