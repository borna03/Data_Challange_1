from config import *
import plot_figures
import clean_data

chosen_airline_id = "20626359"
tweet_count_per_month = plot_figures.tweeted_at_count_month(chosen_airline_id)
reply_count_per_month = plot_figures.replied_at_count_month(chosen_airline_id)

print("Tweet count per month:")
for month_year, count in tweet_count_per_month.items():
    print(f"Month: {month_year}, Count: {count}")

print("Reply count per month:")
for month_year, count in reply_count_per_month.items():
    print(f"Month: {month_year}, Count: {count}")

print(plot_figures.tweeted_at_lang(airlines['VirginAtlantic']['id_str']))
print(plot_figures.responded_to_lang(airlines['VirginAtlantic']['id_str']))