import csv
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['negative', 'neutral', 'positive']

def load_csv_line_by_line(file_path):
    counter = 0
    dct = dict()

    # Read CSV file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            if counter != 0:
                dct[counter] = [line[1], line[10]]  # Store relevant data in dictionary
            counter += 1

    # Process each tweet in the dictionary
    for key, value in dct.items():
        tweet = value[1]
        encoded_tweet = tokenizer(tweet, return_tensors='pt')
        output = model(**encoded_tweet)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)  # Apply softmax to obtain probability distribution

        dct[key].append(labels[np.argmax(scores)])  # Add predicted sentiment label to the dictionary

    # Print the processed data
    for key, value in dct.items():
        print(key, value)

    # Calculate and print the accuracy
    correct_counter = 0
    for key, value in dct.items():
        if value[0] == value[2]:  # Compare actual and predicted labels
            correct_counter += 1

    accuracy = correct_counter / len(dct)
    print(accuracy)

# Specify the file path
file_path = 'test_data/Labled tweets.csv'

# Call the function to load and process the CSV file
load_csv_line_by_line(file_path)
