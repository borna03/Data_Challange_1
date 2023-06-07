from data_load import *
import plot_figures
import data_cleaning
import sentiment_analysis
import topic_classification
import topic_sentiment_plots

chosen_airline_id = "22536055"


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

def analyze_sentiment(chosen_airline_id):
    # sentiment_analysis_from_airline = sentiment_analysis.sentiment_analysis_from_airline(chosen_airline_id)
    # print(sentiment_analysis_from_airline)
    sentiment_analysis_to_airline = sentiment_analysis.sentiment_analysis_to_airline(chosen_airline_id)
    print(sentiment_analysis_to_airline)

# analyze_sentiment(chosen_airline_id)


def analyze_topics(chosen_airline_id):
    topic_sentiment_from_airline = topic_classification.topic_single_sentiment_from_airline(chosen_airline_id)
    print(topic_sentiment_from_airline)
    topic_sentiment_to_airline = topic_classification.topic_single_sentiment_to_airline(chosen_airline_id)
    print(topic_sentiment_to_airline)


# analyze_topics(chosen_airline_id)

def topic_sentiment_plotting():
    topic_sentiment_plots.stacked_bar_chart_topic_sentiment(1)  # Topic stacked bar chart British
    topic_sentiment_plots.stacked_bar_chart_topic_sentiment(2)  # Topic stacked bar char Virgin
    topic_sentiment_plots.pie_chart_topic(1)  # Topic pie chart British
    topic_sentiment_plots.pie_chart_topic(2)  # Topic pie chart Virgin

    # Sentiment stacked bar chart ratio
    topic_sentiment_plots.stacked_bar_chart_sentiment_ratio(topic_sentiment_plots.sentiment_to_airlines)
    # Sentiment multiple bar chart values
    topic_sentiment_plots.multiple_bar_chart_sentiment_numbers(topic_sentiment_plots.sentiment_to_airlines)

topic_sentiment_plotting()
