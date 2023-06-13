from pymongo import UpdateOne, UpdateMany
from data_load import *
import pandas as pd
import heapq
import time


# Function to (re)set a column
def set_new_column(column_name, column_value):
    print('Cleaning/Creating ' + column_name + " column")
    update_operation = {"$set": {column_name: column_value}}
    collection.update_many({}, update_operation)
    print('Done')


# Add the replies to each tweet to the database
def add_replies():
    # Clean/Create the replies column
    set_new_column('replies', [])

    # Initialise needed variables
    tweets = collection.find()
    replies = {}
    bulk_updates = []

    # Start timer
    print('Starting calculation')
    start_time = time.time()

    # Iterate over documents and make a list containing for every tweet the tweets that reply to them
    count = 0
    for tweet in tweets:
        # See if the tweet is a reply and add it to the tweet that it is a reply to
        if tweet['in_reply_to_status_id_str'] is not None:
            replies.setdefault(tweet['in_reply_to_status_id_str'], []).append(
                tweet['id_str'])
        count += 1
        if count % 10000 == 0:
            print('Amount of documents handled: ' + str(count))

    # Define all update operations needed
    for key, value in replies.items():
        filter = {"id_str": {"$eq": key}}
        update_operation = {"$push": {"replies": {"$each": value}}}
        update = UpdateOne(filter, update_operation)
        bulk_updates.append(update)

    # Perform the bulk write operation
    print('Doing the bulk update now')
    start_time_write = time.time()
    collection.bulk_write(bulk_updates)
    end_time = time.time()
    calc_time = start_time_write - start_time
    write_time = end_time - start_time_write
    total_time = end_time - start_time
    print(f"Calc time: {calc_time} seconds")
    print(f"Write time: {write_time} seconds")
    print(f"Total time: {total_time} seconds")
    print('Done')


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
    # Clean/Create the conversation_id column
    set_new_column('conversation_id', 0)

    # Get all tweets
    print('Getting tweets')
    tweets = collection.find()

    # Initialise needed variables
    conversations = {}
    seen_tweets = []
    bulk_updates = []
    conversation_id = 1
    counter = 0
    start_time = time.time()
    tweet_time = start_time

    # Evaluate every tweet and its replies, and add it to the dictionary
    for tweet in tweets:
        # Initialise tracking variables
        active_tweets = tweet['replies'] + [tweet['id_str']]
        unseen_tweets = sorted(active_tweets)
        matching_keys = []

        # Check if we've already handled one of the tweets/replies and get its conversation_id
        for id_str in active_tweets:
            if id_str in seen_tweets:
                keys = [key for key, value in conversations.items() if tweet['id_str'] in value]
                matching_keys.extend(list(set(keys) - set(matching_keys)))
                unseen_tweets.remove(id_str)

        # If we've not seen the tweets before make a new entry and up the conversation_id
        if len(matching_keys) == 0:
            conversations[conversation_id] = active_tweets
            conversation_id += 1

        # If one of the tweets is in another conversation, add all active tweets to that conversation
        elif len(matching_keys) == 1:
            conversations[matching_keys[0]].extend(list(set(active_tweets) - set(conversations[matching_keys[0]])))
        elif len(matching_keys) > 1:
            seen_id = matching_keys[0]
            conversations[seen_id].extend(list(set(active_tweets) - set(conversations[seen_id])))
            for key in matching_keys[1:]:
                conversations[seen_id].extend(list(set(conversations[key]) - set(conversations[seen_id])))
                del conversations[key]

        # Add previously unseen tweets to the seen_tweets list
        seen_tweets = list(heapq.merge(seen_tweets, unseen_tweets))
        counter += 1
        if counter % 10000 == 0:
            current_time = time.time()
            handle_time = current_time - tweet_time
            tweet_time = current_time
            print(f"Batch handle time: {handle_time} seconds")
            print('Handled tweets: ' + str(counter))

    # Define all update operations needed
    print('Adding update operations')
    start_write_time = time.time()
    for key, value in conversations.items():
        filter = {"id_str": {"$in": value}}
        update_operation = {"$set": {"conversation_id": key}}
        update = UpdateMany(filter, update_operation)
        bulk_updates.append(update)

    # Perform bulk update
    print('Performing updates')
    collection.bulk_write(bulk_updates)
    end_time = time.time()
    calc_time = start_write_time - start_time
    write_time = end_time - start_write_time
    total_time = end_time - start_time
    print(f"Calc time: {calc_time} seconds")
    print(f"Write time: {write_time} seconds")
    print(f"Total time: {total_time} seconds")
    print('Done')



