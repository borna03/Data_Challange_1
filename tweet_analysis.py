from data_load import *
import plot_figures
import data_cleaning
import topic_classification

chosen_airline_id = "20626359"

# Function to analyze airline tweets
def analyze_airline_tweets(chosen_airline_id):
    # Get tweet count per month and reply count per month
    tweet_count_per_month = plot_figures.tweeted_at_count_month(chosen_airline_id)
    reply_count_per_month = plot_figures.replied_at_count_month(chosen_airline_id)

    # Print tweet count per month and reply count per month
    print("Tweet count per month:")
    for month_year, count in tweet_count_per_month.items():
        print(f"Month: {month_year}, Count: {count}")

    print("Reply count per month:")
    for month_year, count in reply_count_per_month.items():
        print(f"Month: {month_year}, Count: {count}")

    # Print languages used in tweets by the airline and in replies
    print(plot_figures.tweeted_at_lang(airlines['VirginAtlantic']['id_str']))
    print(plot_figures.responded_to_lang(airlines['VirginAtlantic']['id_str']))

# analyze_airline_tweets(chosen_airline_id)

def analyze_topics(chosen_airline_id):
    topic_single_100 = topic_classification.topic_single_stop(chosen_airline_id, 250)
    topic_multi_100 = topic_classification.topic_multi_stop(chosen_airline_id, 250)
    print(topic_single_100)
    print(topic_multi_100)

analyze_topics(chosen_airline_id)