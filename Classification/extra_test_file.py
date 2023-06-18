import pandas as pd

csv_file = 'ClassificationData/HumanLabeledData.csv'
data = pd.read_csv(csv_file)
negative_reasons = data['negativereason'].unique()

reason_counts = {}
for reason in data['negativereason']:
    if reason in reason_counts:
        reason_counts[reason] += 1
    else:
        reason_counts[reason] = 1

for reason, count in reason_counts.items():
    print(reason, ":", count)