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
GDP= pandas.read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73.csv")
continents.set_index('name', inplace=True)

select = ["Schizophrenia", "Depressive", "Anxiety", "Bipolar", "Eating"]
is2019 = df['Year'] == 2019

data = {}


#Get 2019 GDP's of each country
country_gdp_2019 = pd.Series(GDP['2019'].values, index=GDP['Country Code']).to_dict()

# If you specifically want an array of dictionaries (one per country)
array_country_gdp_2019 = [{k: v} for k, v in country_gdp_2019.items()]

#Get 2019 Anxiety Report
df_2019 = df[df['Year'] == 2019]

# Create a dictionary mapping country codes to their 2019 GDP
country_anxiety_2019 = pd.Series(df_2019.Anxiety.values, index=df_2019.Code).to_dict()

array_country_gdp_2019 = [{k: v} for k, v in country_anxiety_2019.items()]

#Sort both
myKeys = list(country_anxiety_2019.keys())
myKeys.sort()
anxietySorted = {i: country_anxiety_2019[i] for i in myKeys}

myKeys1 = list(country_gdp_2019.keys())
myKeys1.sort()
gdpSorted = {i: country_gdp_2019[i] for i in myKeys1}

def my_filtering_function(pair):
    country_codes = df['Code'].unique()
    country_codes_list = country_codes.tolist()
    wanted_keys = country_codes_list
    key, value = pair
    if key in wanted_keys:
        return True  # keep pair in the filtered dictionary
    else:
        return False  # filter pair out of the dictionary

newGDP= dict(filter(my_filtering_function, gdpSorted.items()))


print(newGDP)
print(anxietySorted)
gdpList = list(newGDP.keys())
anxietyList = list(anxietySorted.keys())
temp3=[]
for element in anxietyList:
    if element not in gdpList:
        temp3.append(element)

print(temp3)
for i in temp3:
    anxietySorted.pop(i)

gdpValues= list(newGDP.values())
anxietyValues= list(anxietySorted.values())


print(len(gdpValues))
print(len(anxietyValues))
plt.scatter(gdpValues,anxietyValues)
plt.show(block=True)





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

