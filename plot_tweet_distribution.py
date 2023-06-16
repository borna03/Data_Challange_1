import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
from data_load import *

def plot_original_tweets_distribution_by_month():
    # Retrieve tweet data from the "data" collection
    tweets = list(collection.find())
    print("Number of tweets retrieved:", len(tweets))

    # Create a DataFrame from the tweets data
    df = pd.DataFrame(tweets)
    print("DataFrame shape:", df.shape)

    # Convert the created_at field to datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    print("Sample created_at value:", df['created_at'].iloc[0])

    # Extract the month from the created_at field
    df['month'] = df['created_at'].dt.month

    # Filter original tweets
    original_tweets = df[(df['in_reply_to_status_id_str'].isnull()) & (df['in_reply_to_user_id_str'].isnull())]
    print("Number of original tweets:", len(original_tweets))

    # Group original tweets by month and count the occurrences
    tweet_counts = original_tweets['month'].value_counts().sort_index()

    # Plot the line chart
    plt.plot(tweet_counts.index, tweet_counts.values)
    plt.xlabel('Month')
    plt.ylabel('Number of Original Tweets')
    plt.title('Distribution of Original Tweets by Month')
    plt.show()


def plot_original_tweets_distribution_by_hour():
    # Retrieve tweet data from the "data" collection
    tweets = list(collection.find())
    print("Number of tweets retrieved:", len(tweets))

    # Create a DataFrame from the tweets data
    df = pd.DataFrame(tweets)
    print("DataFrame shape:", df.shape)

    # Convert the created_at field to datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    print("Sample created_at value:", df['created_at'].iloc[0])

    # Extract the hour from the created_at field
    df['hour'] = df['created_at'].dt.hour

    # Filter original tweets
    original_tweets = df[(df['in_reply_to_status_id_str'].isnull()) & (df['in_reply_to_user_id_str'].isnull())]
    print("Number of original tweets:", len(original_tweets))

    # Group original tweets by hour and count the occurrences
    tweet_counts = original_tweets['hour'].value_counts().sort_index()

    # Calculate the average distribution of tweets per hour
    avg_tweet_counts = tweet_counts.groupby(tweet_counts.index).mean()

    # Plot the line chart
    plt.plot(avg_tweet_counts.index, avg_tweet_counts.values)
    plt.xlabel('Hour')
    plt.ylabel('Average Number of Original Tweets')
    plt.title('Average Distribution of Original Tweets per Hour')
    plt.show()


# Connect to MongoDB
client = MongoClient()
db = client['DBL_test']
collection = db['data']

# Plot distribution of original tweets by month
plot_original_tweets_distribution_by_month()

# Plot average distribution of original tweets by hour
plot_original_tweets_distribution_by_hour()

# Close the MongoDB client connection
client.close()
