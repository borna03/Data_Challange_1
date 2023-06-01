from data_load import *
import pandas as pd


# Function to group all tweets above the current tweet in the same conversation
def check_tweets(tweet_id, conversation_id):
    tweet = tweets[tweets['id_str'] == tweet_id].iloc[0]

    # Check if tweet has already been handled
    if tweet['conversation_id'] is None:
        # Check if the tweet is a root node
        if tweet['in_reply_to_status_id_str'] is not None:

            # Check if the tweet that is replied to, has already been handled
            if tweets[tweets['id_str'] == tweet['in_reply_to_status_id_str']].iloc[0]['conversation_id'] is not None:
                conversation_id = tweets[tweets['id_str'] == tweet['in_reply_to_status_id_str']].iloc[0]['conversation_id']
            else:
                conversation_id = check_tweets(tweet['in_reply_to_status_id_str'], conversation_id)

        tweets.loc[tweets['id_str'] == tweet['id_str'], 'conversation_id'] = conversation_id
    return conversation_id


# Get the tweets from the database and turn them into a dataframe
tweets = pd.DataFrame({'id_str': ['1', '2', '3', '4', '5', '6', '7', '8'], 'in_reply_to_status_id_str': [None, '1', '1', '1', '2', '2', '8', None],
                       'conversation_id': [None, None, None, None, None, None, None, None]})

# Run over all tweets in new collection
counter = 1
for x in tweets.values:
    output = check_tweets(x[0], str(counter))
    counter += 1

#tweet = tweets[tweets['id_str'] == '1'].iloc[0]
print(tweets)
