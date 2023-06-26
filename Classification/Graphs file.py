import matplotlib.pyplot as plt
import squarify
import numpy as np
from Classification_Visualisation import *

print(british_airways_topic_counts)
print(virgin_atlantic_topic_counts)

topics = [item for item in virgin_atlantic_topic_counts]

sentiments = ['neutral', 'positive', 'negative', 'uncertain']

def multiple_bar_plot(data, labels, title, save_file_name):
    N = len(labels)
    ind = np.arange(N)
    width = 0.16

    # Get data for all sentiments for all topics
    neu_count = []
    pos_count = []
    neg_count = []
    unc_count = []
    for topic, sentiment in data.items():
        for sentiment, count in sentiment.items():
            if sentiment == 'neutral':
                neu_count.append(count)
            if sentiment == 'positive':
                pos_count.append(count)
            if sentiment == 'negative':
                neg_count.append(count)
            if sentiment == 'uncertain':
                unc_count.append(count)

    # Create a bar for each sentiment
    bar_neu = plt.bar(ind, neu_count, width, color='cornflowerblue')
    bar_pos = plt.bar(ind+width, pos_count, width, color='mediumspringgreen')
    bar_neg = plt.bar(ind+width*2, neg_count, width, color='orangered')
    bar_unc = plt.bar(ind+width*3, unc_count, width, color='dimgrey')

    # Set the x-axis labels and tick positions
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.xticks(ind + width * 1.5, labels, rotation=45, ha='right')
    plt.title(title)

    # Add a legend
    plt.legend((bar_neu, bar_pos, bar_neg, bar_unc), sentiments)

    # Display the chart
    plt.tight_layout()
    plt.savefig(f'plots/{save_file_name}', bbox_inches='tight')
    plt.show()

multiple_bar_plot(virgin_atlantic_topic_counts , topics, "Virgin Atlantic Sentiment Analysis", 'Virgin')
multiple_bar_plot(british_airways_topic_counts , topics,"British Airways Sentiment Analysis", 'British')

def tree_map_chart(data1, data2):
    # Extract data from the first dictionary
    labels1 = list(data1.keys())
    sizes1 = list(data1.values())

    # Extract data from the second dictionary
    labels2 = list(data2.keys())
    sizes2 = list(data2.values())

    # Sort the data by size in descending order
    sorted_indices1 = np.argsort(sizes1)[::-1]
    labels1 = [labels1[i] for i in sorted_indices1]
    sizes1 = [sizes1[i] for i in sorted_indices1]

    sorted_indices2 = np.argsort(sizes2)[::-1]
    labels2 = [labels2[i] for i in sorted_indices2]
    sizes2 = [sizes2[i] for i in sorted_indices2]

    # Generate tree map rectangles for both charts
    rects1 = squarify.normalize_sizes(sizes1, 10, 10)
    rects2 = squarify.normalize_sizes(sizes2, 10, 10)

    # Create a color scale for the tree map charts
    color_scale1 = plt.cm.Blues(np.linspace(1, 0.2, len(data1)))
    color_scale2 = plt.cm.Blues(np.linspace(1, 0.2, len(data2)))

    # Create the tree map figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    # Plot the first tree map chart
    axes[0].title.set_text('Tree Map Chart 1')
    squarify.plot(sizes=sizes1, label=labels1, color=color_scale1, alpha=0.8, ax=axes[0], text_kwargs={'fontsize': 'small'})
    axes[0].axis('off')

    # Plot the second tree map chart
    axes[1].title.set_text('Tree Map Chart 2')
    squarify.plot(sizes=sizes2, label=labels2, color=color_scale2, alpha=0.8, ax=axes[1], text_kwargs={'fontsize': 'small'})
    axes[1].axis('off')

    # Adjust the spacing between the subplots
    plt.tight_layout()

    # Show the chart
    plt.show()


def count_tweets_per_topic(data):
    tweet_counts = {}

    for tweet in data:
        topic = tweet['topic']
        count = tweet['count']

        if topic in tweet_counts:
            tweet_counts[topic] += count
        else:
            tweet_counts[topic] = count

    return tweet_counts


# tree_map_chart(count_tweets_per_topic(british_airways_data), count_tweets_per_topic(virgin_atlantic_data))
