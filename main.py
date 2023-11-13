import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

dataset = pd.read_csv("em-9EhjTEemU7w7-EFnPcg_7aa34fc018d311e980c2cb6467517117_happyscore_income.csv")

dataset.sort_values('median_income', inplace=True)

highest_median_income = dataset["median_income"].iloc[-1] *0.6

highest_median_income = dataset[dataset["median_income"] > highest_median_income]

median_income = dataset['median_income']
avg_satisfaction = dataset['avg_satisfaction']

median_income_mean = np.mean(highest_median_income['median_income'])
avg_satisfaction_mean = np.mean(dataset['avg_satisfaction'])

higher_median_income = median_income.max()
print(higher_median_income)

join_columns = np.column_stack((avg_satisfaction, median_income))



kmeans_resul = KMeans(n_clusters=3).fit(join_columns)

clusters = kmeans_resul.cluster_centers_
plt.scatter(avg_satisfaction, median_income)
plt.scatter(clusters[:, 0], clusters[:, 1], s=1000, alpha=0.6)

plt.xlabel("average_satisfaction")
plt.ylabel("median_income")

plt.show()


print("Columns:")
for column in dataset.columns:
    print(column)