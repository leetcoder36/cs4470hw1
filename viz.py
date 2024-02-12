import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


df = pandas.read_csv("mental-illness_data.csv")

select = ["Schizophrenia", "Depressive", "Anxiety", "Bipolar", "Eating"]
is2019 = df['Year'] == 2019

data = {}
for disorder in select:
    mean = df.loc[is2019, disorder].mean()
    data[disorder] = mean

custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
g1 = sns.barplot(data, palette=custom_palette).set_title("Mean Prevalence of Mental Illnesses in 2019")
plt.xlabel("Mental Disorder Type")
plt.ylabel("Mean Prevalence Across All Countries (%)")
plt.savefig("Graphs/disorderMeans2019.png")

fig, ax = plt.subplots(2, 5, figsize=(16, 6))
palette = sns.color_palette("tab10", 10)
combinations = list(combinations(select, 2))
counter = 0
for combination in combinations:
    x = df.loc[is2019, combination[0]]
    y = df.loc[is2019, combination[1]]
    if counter <= 4:
        sns.scatterplot(x=x, y=y, ax=ax[0, counter], color=palette[counter])
    else:
        sns.scatterplot(x=x, y=y, ax=ax[1, 5 - counter], color=palette[counter])
    counter += 1

fig.suptitle("Correlations Between Prevalence of Mental Disorders Across All Countries in 2019")
plt.tight_layout()
plt.savefig("Graphs/disorderCorrelations.png")



