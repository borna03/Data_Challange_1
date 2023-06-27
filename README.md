# DBL-DC 12

## Introduction
Welcome to DBL-DC 12! This project aims to analyze and visualize Twitter data for sentiment and topic classification. Follow the instructions below to get started.

## Getting Started
Before running the code, please ensure that you have installed all the required packages listed in the `requirements.txt` file. Additionally, make sure MongoDB is properly set up and configure the related parameters in the code accordingly.

## Data Loading
To load the data, follow these steps:

1. Open the `clean_data.py` file and execute the `clean()` function to clean the data.
2. Run the `clean2()` function in the same file to perform additional cleaning steps.
3. Proceed to the `conv_collection.py` file and run it. Follow the prompts to set up a collection containing only conversations.
4. In the `creat_conversations.py` file, execute the `add_conversations()` and `add_levels()` functions in that order.

## Sentiment Analysis and Topic Classification
To perform sentiment analysis and topic classification, follow these steps:

1. Execute the `sentimentanalysis.py` file.
2. Open the `tweet_analysis.py` file. It contains a function that combines both sentiment analysis and topic classification. You can find the individual functions in the `sentiment_analysis.py` and `topic_classification.py` files respectively. Run the desired function(s) accordingly.

## Plotting
To plot sentiment over time, run the following functions from the `plot_tweet_distribution.py` file:
- `plot_sentiment_count_virgin`
- `plot_sentiment_count_british`
- `plot_sentiment_count_others`

Ensure that you provide the corresponding dictionaries saved in the `data_for_plots4.py` file as parameters.

For other figures, execute the respective files according to your requirements.

## Project Status
The project is completed and fully functional.
