from pymongo import UpdateOne, UpdateMany
from data_load import *
import pandas as pd
import heapq
import time


# Compilation function
def add_conversations():
    absolute_start = time.time()
    start_time = time.time()
    add_replies()
    end_time = time.time()
    print(f"add_replies finished in {round((end_time-start_time)/60, 2)} minutes")
    start_time = end_time
    general_conversations()
    end_time = time.time()
    print(f"general_conversations finished in {round((end_time-start_time)/60, 2)} minutes")
    start_time = end_time
    subdivide_conversations()
    end_time = time.time()
    print(f"subdivide_conversations finished in {round((end_time-start_time)/60, 2)} minutes")
    start_time = end_time
    subdivide_large_conversation(109384)
    subdivide_large_conversation(289444)
    end_time = time.time()
    print(f"subdivide_large_conversation finished in {round((end_time-start_time)/60, 2)} minutes")
    print(f"Total runtime: {round((end_time-absolute_start)/60, 2)} minutes")


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
    print('Getting tweets')
    tweets = collection.find()
    replies = {}
    bulk_updates = []
    count = 0

    # Iterate over documents and make a list containing for every tweet the tweets that reply to them
    print('Starting calculation')
    for tweet in tweets:
        # See if the tweet is a reply and add it to the tweet that it is a reply to
        if tweet['in_reply_to_status_id_str'] is not None:
            replies.setdefault(tweet['in_reply_to_status_id_str'], []).append(
                tweet['id_str'])
        count += 1
        if count % 100000 == 0:
            print('Amount of documents handled: ' + str(count))

    # Define all update operations needed
    print('Defining update operations')
    for key, value in replies.items():
        filter = {"id_str": {"$eq": key}}
        update_operation = {"$push": {"replies": {"$each": value}}}
        update = UpdateOne(filter, update_operation)
        bulk_updates.append(update)

    # Perform the bulk write operation
    print('Doing the bulk update now')
    collection.bulk_write(bulk_updates)
    print('Done')


# For very large conversations
def subdivide_large_conversation(conversation_id):
    print('Getting conversations')
    conversation = collection.find({'conversation_id': conversation_id})
    convo = []
    print('Assembling convo')
    for tweet in conversation:
        convo.append(tweet)

    print('Finding sub-conversations')
    bulk_updates = subdivide_conversation(convo)
    print('Updating database')
    collection.bulk_write(bulk_updates)


# Function to create the sub-conversations
def subdivide_conversations():
    # Create/Clean the sub-conversation_id column
    start_time = time.time()
    set_new_column('sub-conversation_id', [])

    # Gather all conversations grouped by conversation_id
    pipeline = [
        {"$match": {"conversation_id": {"$nin": [0, 109384, 289444]}}},
        {"$group": {"_id": "$conversation_id", "tweets": {"$push": "$$ROOT"}}}
    ]

    print('Retrieving all conversations')
    conversations = collection.aggregate(pipeline)

    # Initialise bulk_updates
    bulk_updates = []

    # Loop over all general conversations
    print('Subdividing conversations')
    start_time = time.time()
    counter = 0
    convo_time = time.time()
    for convo in conversations:
        bulk_updates.extend(subdivide_conversation(convo['tweets']))
        counter += 1
        if counter % 50000 == 0:
            print(f"Finished {format(counter, ',')}/372,362 conversations")
            print(f"Batch completed in {round(time.time() - convo_time, 2)} seconds")
            convo_time = time.time()

    # Perform the update
    print('Performing the update')
    collection.bulk_write(bulk_updates)
    end_time = time.time()
    print(f"Finished updates in {round(end_time-convo_time, 2)} seconds")
    print(f"Total process time: {round((end_time-start_time)/60, 2)} minutes")


# Finds the sub-conversations in a conversation and outputs the update operation
def subdivide_conversation(convo):
    # Get the levels and chain of every leaf tweet in the conversation
    tweets = {tweet['id_str']: tweet['in_reply_to_status_id_str'] for tweet in convo}
    leaf_nodes = get_convo_info(convo, tweets)
    subconversation_id = 1
    bulk_updates = []

    # Loop over all tweets in the conversation
    for leaf_id, leaf in leaf_nodes.items():
        # Check if leaf is handled
        if leaf['handled'] is False:
            # Add all tweets in the tweet chain
            subconvo_tweets = leaf['chain']

            # Add all tweets on the same level that reply to the same tweet
            for key, value in leaf_nodes.items():
                if value['level'] == leaf['level'] and tweets[leaf_id] == tweets[key] and leaf_id != key:
                    subconvo_tweets.append(key)
                    value['handled'] = True

            # Add the update operation to bulk_updates
            filter = {"id_str": {'$in': subconvo_tweets}}
            update_operation = {"$push": {"sub-conversation_id": subconversation_id}}
            update = UpdateMany(filter, update_operation)
            bulk_updates.append(update)
            subconversation_id += 1
    return bulk_updates


# Gets general information about all tweets in a conversation
def get_convo_info(conversation, tweets):
    info = {}
    for tweet in conversation:
        # Only handle if it is a leaf node
        if not tweet['replies']:
            info[tweet['id_str']] = {}
            info[tweet['id_str']]['level'] = levels_count(tweets, tweet["id_str"], 0)
            info[tweet['id_str']]['chain'] = get_tweet_chain(tweets, tweet['id_str'])
            info[tweet['id_str']]['handled'] = False
    return info


# Get all tweets in a chain (given a leaf node)
def get_tweet_chain(tweets, id_str):
    if id_str is None:
        return []
    else:
        chain = [id_str]
        chain.extend(get_tweet_chain(tweets, tweets.get(id_str)))
        return chain


# Function that determines the level of all tweets in a conversation
def levels_count(tweet_chain, id_str, level):
    if tweet_chain[id_str] is None:
        return level + 1
    else:
        return levels_count(tweet_chain, tweet_chain[id_str], level) + 1


# Function for defining general conversations
def general_conversations():
    # Creating/Cleaning conversation_id column
    set_new_column('conversation_id', 0)

    # Getting all roots
    print('Getting roots')
    root_tweets = collection.find({'in_reply_to_status_id_str': None})

    # Initialising needed variables
    roots = 372362
    conversations = {}
    start_time = time.time()
    convo_start_time = start_time
    counter = 0
    total = 0

    # Getting general conversation for every root
    print('Getting conversations (batches of 10.000)')
    for root in root_tweets:
        # Getting general conversation
        conversations[root['id_str']] = build_conversation_chain(root, [])
        counter += 1
        if counter % 10000 == 0:
            convo_end_time = time.time()
            convo_time = round((convo_end_time - convo_start_time), 2)
            convo_start_time = convo_end_time
            total += convo_time
            print(f"Conversations found: {format(counter, ',')} out of {format(roots, ',')} possible conversations")
            print(f"Batch time: {convo_time} seconds")
            print(f"Estimated time left for conversation finding: "
                  f"{round((((roots-counter)/10000)*(total/(counter/10000)))/60, 2)} minutes")
            print("-----------------------")

    print('Creating update operations')
    conversation_id = 1
    bulk_updates = []
    for root, replies in conversations.items():
        # Convert to a list of id's
        id_list = [root] + [reply for reply in replies]

        # Add an update operation to bulk_updates
        filter = {"id_str": {"$in": id_list}}
        update_operation = {"$set": {"conversation_id": conversation_id}}
        update = UpdateMany(filter, update_operation)
        bulk_updates.append(update)

        # Up the conversation_id
        conversation_id += 1

    end_time = time.time()
    print(f"Found {conversation_id} conversations in {round(((end_time - start_time) / 60), 2)} minutes")
    print('Performing update operations')
    collection.bulk_write(bulk_updates)
    print(f'Finish entire process in {round(((time.time() - start_time)/60), 2)} minutes')


# Function for recursively finding general conversations
def build_conversation_chain(tweet, convo_chain):
    # For every reply get the replies and add them to the chain
    for reply_id in tweet['replies']:
        reply = collection.find_one({'id_str': reply_id})
        convo_chain.append(reply_id)
        build_conversation_chain(reply, convo_chain)
    return convo_chain

