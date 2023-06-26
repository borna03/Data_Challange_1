import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["DBL_data"]
# collection = db["final_data"]
db = client["DBL_clean"]
collection = db["conv_sent_full_test"]

ids = ["20626359", "18332190"]
british_airways_tweets = []
virgin_atlantic_tweets = []

for id_str in ids:
    query = {
        '$or': [
            {'in_reply_to_user_id_str': id_str},
            {'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}}
        ]
    }
    if id_str == "20626359":
        british_airways_tweets = list(collection.find(query))
    if id_str == "18332190":
        virgin_atlantic_tweets = list(collection.find(query))

client.close()
print(len(british_airways_tweets), len(virgin_atlantic_tweets))


def extract_tweet_data(tweet):
    topic_specific = tweet.get("topic_specific")
    sentiment = tweet.get("sentiment")
    return (topic_specific, sentiment)


# Extract data from British Airways tweets
british_airways_data = [extract_tweet_data(tweet) for tweet in british_airways_tweets]

# Extract data from Virgin Atlantic tweets
virgin_atlantic_data = [extract_tweet_data(tweet) for tweet in virgin_atlantic_tweets]

# Create a dictionary to store the topic counts for each airline
british_airways_topic_counts = {}
virgin_atlantic_topic_counts = {}


# Function to update the topic counts dictionary
def update_topic_counts(topic_counts, topic, sentiment):
    if topic in topic_counts:
        if sentiment in topic_counts[topic]:
            topic_counts[topic][sentiment] += 1
        else:
            topic_counts[topic][sentiment] = 1
    else:
        topic_counts[topic] = {sentiment: 1}


# Count the number of sentiment scores per topic category for British Airways
for topic, sentiment in british_airways_data:
    update_topic_counts(british_airways_topic_counts, topic, sentiment)

# Count the number of sentiment scores per topic category for Virgin Atlantic
for topic, sentiment in virgin_atlantic_data:
    update_topic_counts(virgin_atlantic_topic_counts, topic, sentiment)

desired_order_sentiment = ['neutral', 'positive', 'negative', 'uncertain']

# Function to save topic counts to a text file
def save_topic_counts(file_path, topic_counts):
    with open(file_path, "w") as file:
        for topic, sentiments in topic_counts.items():
            file.write(topic + "\n")
            a = dict(sentiments.items())
            for sentiment in desired_order_sentiment:
                count = a[sentiment]
                file.write(f"{sentiment}: {count}\n")
            file.write("\n")

# Re-order the dictionaries
desired_order_list = ['Customer Service','On-Flight Experience', 'Delays & Cancellations', 'Baggage',
                      'Claims & Refunds', 'Financial (Prices, fees, air-miles)', '(Online) Booking & Seats',  'Security, gates & Long Lines',
                      'Appreciation Messages', 'General Complaints & Hate Messages', 'Undefined / Unrelated']
british_airways_topic_counts = {k: british_airways_topic_counts[k] for k in desired_order_list}
virgin_atlantic_topic_counts = {k: virgin_atlantic_topic_counts[k] for k in desired_order_list}
print(british_airways_topic_counts)
print(virgin_atlantic_topic_counts)

# Save the topic counts for British Airways to a text file
british_airways_file_path = "topic_counts_british_airways.txt"
save_topic_counts(british_airways_file_path, british_airways_topic_counts)
print("Topic counts for British Airways saved to", british_airways_file_path)

# Save the topic counts for Virgin Atlantic to a text file
virgin_atlantic_file_path = "topic_counts_virgin_atlantic.txt"
save_topic_counts(virgin_atlantic_file_path, virgin_atlantic_topic_counts)
print("Topic counts for Virgin Atlantic saved to", virgin_atlantic_file_path)

# Function to calculate the total counts for each sentiment score across all topics
def calculate_total_counts(topic_counts):
    total_counts = {'uncertain': 0, 'negative': 0, 'positive': 0, 'neutral': 0}
    for topic, sentiments in topic_counts.items():
        for sentiment, count in sentiments.items():
            total_counts[sentiment] += count
    return total_counts

# Calculate the total counts for each airline
british_airways_total_counts = calculate_total_counts(british_airways_topic_counts)
virgin_atlantic_total_counts = calculate_total_counts(virgin_atlantic_topic_counts)

print(british_airways_total_counts)
print(virgin_atlantic_total_counts)
print('\n')

