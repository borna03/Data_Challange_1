import matplotlib.pyplot as plt
import squarify
import numpy as np

virgin_atlantic_data = [
    {'topic': 'Undefined / Unrelated', 'sentiment': 'neutral', 'count': 21750},
    {'topic': 'Undefined / Unrelated', 'sentiment': 'positive', 'count': 9759},
    {'topic': 'Undefined / Unrelated', 'sentiment': 'negative', 'count': 8999},
    {'topic': 'Undefined / Unrelated', 'sentiment': 'uncertain', 'count': 3992},
    {'topic': 'On-Flight Experience', 'sentiment': 'neutral', 'count': 12406},
    {'topic': 'On-Flight Experience', 'sentiment': 'positive', 'count': 15877},
    {'topic': 'On-Flight Experience', 'sentiment': 'negative', 'count': 11682},
    {'topic': 'On-Flight Experience', 'sentiment': 'uncertain', 'count': 3351},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'neutral', 'count': 18360},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'positive', 'count': 5888},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'negative', 'count': 12315},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'uncertain', 'count': 3141},
    {'topic': 'Customer Service', 'sentiment': 'neutral', 'count': 19815},
    {'topic': 'Customer Service', 'sentiment': 'positive', 'count': 7457},
    {'topic': 'Customer Service', 'sentiment': 'negative', 'count': 21318},
    {'topic': 'Customer Service', 'sentiment': 'uncertain', 'count': 3982},
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'neutral', 'count': 3660},
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'positive', 'count': 2642},
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'negative', 'count': 5552},
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'uncertain', 'count': 1287},
    {'topic': 'Appreciation Messages', 'sentiment': 'neutral', 'count': 2302},
    {'topic': 'Appreciation Messages', 'sentiment': 'positive', 'count': 10752},
    {'topic': 'Appreciation Messages', 'sentiment': 'negative', 'count': 1159},
    {'topic': 'Appreciation Messages', 'sentiment': 'uncertain', 'count': 784},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'neutral', 'count': 11290},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'positive', 'count': 1697},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'negative', 'count': 9057},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'uncertain', 'count': 1119},
    {'topic': 'Claims & Refunds', 'sentiment': 'neutral', 'count': 11442},
    {'topic': 'Claims & Refunds', 'sentiment': 'positive', 'count': 1038},
    {'topic': 'Claims & Refunds', 'sentiment': 'negative', 'count': 6561},
    {'topic': 'Claims & Refunds', 'sentiment': 'uncertain', 'count': 1028},
    {'topic': 'Baggage', 'sentiment': 'neutral', 'count': 5575},
    {'topic': 'Baggage', 'sentiment': 'positive', 'count': 1091},
    {'topic': 'Baggage', 'sentiment': 'negative', 'count': 4924},
    {'topic': 'Baggage', 'sentiment': 'uncertain', 'count': 864},
    {'topic': 'Delays & Cancellations', 'sentiment': 'neutral', 'count': 12215},
    {'topic': 'Delays & Cancellations', 'sentiment': 'positive', 'count': 2621},
    {'topic': 'Delays & Cancellations', 'sentiment': 'negative', 'count': 14273},
    {'topic': 'Delays & Cancellations', 'sentiment': 'uncertain', 'count': 1741},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'neutral', 'count': 1154},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'positive', 'count': 320},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'negative', 'count': 1181},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'uncertain', 'count': 219}
]

british_airways_data = [
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'neutral', 'count': 1147},
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'positive', 'count': 1444},
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'negative', 'count': 1066},
    {'topic': 'General Complaints & Hate Messages', 'sentiment': 'uncertain', 'count': 339},
    {'topic': 'Undefined / Unrelated', 'sentiment': 'neutral', 'count': 10951},
    {'topic': 'Undefined / Unrelated', 'sentiment': 'positive', 'count': 6318},
    {'topic': 'Undefined / Unrelated', 'sentiment': 'negative', 'count': 2050},
    {'topic': 'Undefined / Unrelated', 'sentiment': 'uncertain', 'count': 1183},
    {'topic': 'On-Flight Experience', 'sentiment': 'neutral', 'count': 4475},
    {'topic': 'On-Flight Experience', 'sentiment': 'positive', 'count': 9311},
    {'topic': 'On-Flight Experience', 'sentiment': 'negative', 'count': 2324},
    {'topic': 'On-Flight Experience', 'sentiment': 'uncertain', 'count': 981},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'neutral', 'count': 5092},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'positive', 'count': 3203},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'negative', 'count': 2385},
    {'topic': 'Financial (Prices, fees, air-miles)', 'sentiment': 'uncertain', 'count': 820},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'neutral', 'count': 2722},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'positive', 'count': 809},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'negative', 'count': 1409},
    {'topic': '(Online) Booking & Seats', 'sentiment': 'uncertain', 'count': 275},
    {'topic': 'Customer Service', 'sentiment': 'neutral', 'count': 4947},
    {'topic': 'Customer Service', 'sentiment': 'positive', 'count': 3970},
    {'topic': 'Customer Service', 'sentiment': 'negative', 'count': 3606},
    {'topic': 'Customer Service', 'sentiment': 'uncertain', 'count': 1013},
    {'topic': 'Appreciation Messages', 'sentiment': 'neutral', 'count': 626},
    {'topic': 'Appreciation Messages', 'sentiment': 'positive', 'count': 3967},
    {'topic': 'Appreciation Messages', 'sentiment': 'negative', 'count': 228},
    {'topic': 'Appreciation Messages', 'sentiment': 'uncertain', 'count': 178},
    {'topic': 'Delays & Cancellations', 'sentiment': 'neutral', 'count': 2925},
    {'topic': 'Delays & Cancellations', 'sentiment': 'positive', 'count': 1194},
    {'topic': 'Delays & Cancellations', 'sentiment': 'negative', 'count': 2027},
    {'topic': 'Delays & Cancellations', 'sentiment': 'uncertain', 'count': 348},
    {'topic': 'Baggage', 'sentiment': 'neutral', 'count': 1230},
    {'topic': 'Baggage', 'sentiment': 'positive', 'count': 459},
    {'topic': 'Baggage', 'sentiment': 'negative', 'count': 699},
    {'topic': 'Baggage', 'sentiment': 'uncertain', 'count': 184},
    {'topic': 'Claims & Refunds', 'sentiment': 'neutral', 'count': 1240},
    {'topic': 'Claims & Refunds', 'sentiment': 'positive', 'count': 332},
    {'topic': 'Claims & Refunds', 'sentiment': 'negative', 'count': 669},
    {'topic': 'Claims & Refunds', 'sentiment': 'uncertain', 'count': 147},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'neutral', 'count': 238},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'positive', 'count': 127},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'negative', 'count': 176},
    {'topic': 'Security, gates & Long Lines', 'sentiment': 'uncertain', 'count': 48}
]

topics = ['Undefined / Unrelated', 'On-Flight Experience', 'Financial (Prices, fees, air-miles)', 'Customer Service',
          'General Complaints & Hate Messages', 'Appreciation Messages', '(Online) Booking & Seats', 'Claims & Refunds',
          'Baggage', 'Delays & Cancellations', 'Security, gates & Long Lines']

topics2 = ['General Complaints & Hate Messages', 'Undefined / Unrelated', 'On-Flight Experience',
           'Financial (Prices, fees, air-miles)', '(Online) Booking & Seats', 'Customer Service',
           'Appreciation Messages', 'Delays & Cancellations', 'Baggage', 'Claims & Refunds',
           'Security, gates & Long Lines']

sentiments = ['neutral', 'positive', 'negative', 'uncertain']


def multiple_bar_plot(data, labels, title):
    # Set the position of the bars on the x-axis
    x_pos = list(range(len(labels)))

    # Set the width of each bar
    bar_width = 0.1

    # Create a bar for each sentiment
    for i, sentiment in enumerate(sentiments):
        counts = [item['count'] for item in data if item['sentiment'] == sentiment]
        plt.bar([x + i * bar_width for x in x_pos], counts, width=bar_width, label=sentiment)

    # Set the x-axis labels and tick positions
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.xticks(x_pos, labels, rotation=45, ha='right')

    # Add a legend
    plt.legend()
    plt.title(title)

    # Display the chart
    plt.tight_layout()
    plt.show()


multiple_bar_plot(virgin_atlantic_data , topics , "Virgin Atlantic Sentiment Analysis")
multiple_bar_plot(british_airways_data , topics2,"British Airways Sentiment Analysis")

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


tree_map_chart(count_tweets_per_topic(british_airways_data), count_tweets_per_topic(virgin_atlantic_data))
