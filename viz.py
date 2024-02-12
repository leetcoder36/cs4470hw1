import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import plotly.express as px
import plotly.io as pio


df = pandas.read_csv("mental-illness_data.csv")
continents = pandas.read_csv("continents2.csv")
continents.set_index('name', inplace=True)

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

fig.suptitle("Correlations Between Prevalence of Mental Disorders Across All Countries in 2019 (%)")
plt.tight_layout()
plt.savefig("Graphs/disorderCorrelations.png")
plt.clf()

fig, ax = plt.subplots(1, 5, figsize=(16, 6))
counter = 0
for disorder in select:
    means_per_year = {}
    for year in df["Year"]:
        if year % 2 == 0:
            mean = df.loc[df["Year"] == year, disorder].mean()
            means_per_year[year] = mean

    plot = sns.lineplot(means_per_year, ax=ax[counter])
    ax[counter].set_title(disorder)
    plot.set(xlabel='Year', ylabel='Mean Percentage of Country Population With ' + disorder + " Disorder (%)")
    counter += 1

fig.suptitle("Mean Prevalence of Disorders Throughout All Countries Over Time")
plt.tight_layout()
plt.savefig("Graphs/meansOverTime.png")
plt.clf()

for disorder in select:
    fig = px.choropleth(df, locations='Code', color=disorder, hover_name='Entity',
                        projection='natural earth',
                        title=disorder + ' Prevalence as Percentage of Population by Country',
                        animation_frame='Year')
    pio.write_html(fig, file="Graphs/" + disorder + 'Map.html', auto_open=False)

select1 = select[:2]
select2 = select[2:]

fig, ax = plt.subplots(1, len(select1), figsize=(20, 6))
counter = 0
for disorder in select1:
    top5 = df.loc[is2019].nlargest(5, disorder)
    adjusted = top5.loc[:, ['Entity', disorder]]
    print(adjusted)
    plot = sns.barplot(x=adjusted["Entity"], y=adjusted[disorder], ax=ax[counter])
    plot.set(ylabel="% of Population Suffering From " + " Disorder (%)", xlabel="Country", title=disorder)
    counter += 1

fig.suptitle("Top 5 Countries With Highest Prevalence of Each Disorder in 2019 (1)")
plt.savefig("Graphs/highestInstances1.png")

fig, ax = plt.subplots(1, len(select2), figsize=(20, 6))
counter = 0
for disorder in select2:
    top5 = df.loc[is2019].nlargest(5, disorder)
    adjusted = top5.loc[:, ['Entity', disorder]]
    print(adjusted)
    plot = sns.barplot(x=adjusted["Entity"], y=adjusted[disorder], ax=ax[counter])
    plot.set(ylabel="% of Population Suffering From " + " Disorder (%)", xlabel="Country", title=disorder)
    counter += 1

fig.suptitle("Top 5 Countries With Highest Prevalence of Each Disorder in 2019 (2)")
plt.savefig("Graphs/highestInstances2.png")
