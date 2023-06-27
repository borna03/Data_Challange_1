# DBL-DC 12 



## Getting started
Make sure you install required package using the requirements.txt
Make sure MongoDB is setup and check related parameters in the code.

## Load Data
From the clean_data.py first run `clean()` then run `clean2()`.
Subsequently run the conv_collection.py file following the prompts to setup a collection containing only conversations.
From the `creat_conversations.py` run `add_conversations()` and `add_levels()` in that order.
Now run in this order: `sentimentanalysis.py`, `
In `tweet_analysis.py`, you have a function to run both of the functions you can find in `sentiment_analysis.py` for the sentiment. You can also find for the (first) topic classification in `tweet_analysis.py`, to run both function in `topic_classification.py`.
To plot the plots for sentiment over time, run the `plot_sentiment_count_virgin`, `plot_sentiment_count_british` and `plot_sentiment_count_others` from `plot_tweet_distribution.py`.
Use the corresponding dictionaries that have been pre saved in the `data_for_plots4.py` when running as parameters.
To plot the other figures execute the corresponding files.

## Project status
Done
