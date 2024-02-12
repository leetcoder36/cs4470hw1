'''
data visualizations - in class Spring 2024

'''
'''
Visualization and correlation only

This dataset is originally from the National Institute of Diabetes
and Digestive and Kidney Diseases. The objective of the dataset is
to diagnostically predict whether or not a patient has diabetes,
based on certain diagnostic measurements included in the dataset.
Several constraints were placed on the selection of these instances
from a larger database. In particular, all patients here are females
at least 21 years old of Pima Indian heritage.

Glucose:  Plasma glucose concentration a 2 hours in an oral
glucose tolerance test

BloodPressure: Diastolic (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)
`
Insulin: 2-Hour serum insulin (mu U/ml)

BMI: (weight in Kg/height in m)^2

DiabetesPedigreeFunction: Diabetes pedigree function

Age:

Outcome: 0 - no diabetes, 1 - diabetes,  268/768 are 1, rest 0.

[PS added PregCateg and AgeCategory as categorical variables]

'''

##  - IMPORT LIBRARIES  ###############

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



##  - READ DATA INTO DF ###############

df = pd.read_csv("diabetes.csv")
df = df.rename(columns = {'DiabetesPedigreeFunction':'DPF'})
df.shape
df.head
df


def description(df):
    '''
    Handy function to get an overall description of a dataset
    '''
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    
    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())

    # creating an output df    
    output = pd.DataFrame({
        'variable': variables, 
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing
    })          
    return output


df.describe()
df['BMI'].describe()

'''
#   looking more closely at min values.  Are there any zeros?
#   do zeros make any sense?

# 
# Glucose, BloodPressure, SkinThickness, Insulin, BMI.

'''
### replacing zeros with NaN

df_copy = df.copy(deep = True)

df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
                            df_copy[['Glucose','BloodPressure','SkinThickness',\
                                     'Insulin','BMI']].replace(0,np.NaN)


print(df.isnull().sum())

print(df_copy.isnull().sum())

###  IMPUTING VALUES FOR NAN


df_copy[['Glucose','BloodPressure','SkinThickness',\
                                     'Insulin','BMI']].hist(figsize = (20, 20))
         

plt.show()


# Imputing values for NaNs
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace = True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace = True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace = True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace = True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace = True)

df_copy.isnull().sum()

df_orig = df.copy(deep = True)

df = df_copy

# Seaborn (sns) visualizations built on top of matplotlib, 

SelectFeatures = ['BloodPressure','BMI']
SelectFeatures2 = ['Insulin', 'BloodPressure','BMI']

### DISTPLOT: hist bool, kde bool, bins hist bins.

#hist = False for no histogram, try kde = True
for i in SelectFeatures:
    sns.distplot(x = df[i], hist= True, kde = False)\
                .set(title='Distplot '+i)
    plt.show()



### KDEPLOT ###

'''
A kernel density estimate (KDE) plot is a method for visualizing the
distribution of observations in a dataset, analogous to a histogram.
Diff: KDE represents the data using a continuous probability density curve
in one or more dimensions.
'''

# y = for horizontal plot, notice hue, palette has a default
for i in SelectFeatures2:
    sns.kdeplot(x = df[i], hue = df['Outcome'], palette = 'PRGn').set(title='KDE plot '+i)  
    plt.show()

# try palletes - Accent_r, BrBG, BuPu, CMRmap, Greys, Greys_r, PRGn, (all work)   

### BOXPLOTS ###
'''
box plot or boxplot nicely demonstrate the locality, spread and
skewness groups of numerical data through their
quartiles. In addition to the box on a box plot, there can be lines
(which are called whiskers)
extending from the box indicating variability outside the upper and lower
quartiles,thus, the plot is also called the box-and-whisker plot.
median, lower and upper quartiles, min and max excluding outliers.
for min and max IQR (inter quartile range) * 1.5.

The box plot helps identify the 25th and 75th percentiles better than
the histogram, while the histogram helps you see the overall shape
of your data better than the box plot.
'''

for i in SelectFeatures:
    # can set notch = True
    sns.boxplot(y = i, data = df,notch = False, # y to x for horizontal
           palette = 'CMRmap').set(title='boxplot '+i)  #lightblue
    plt.show()


#### BOXENPLOT #  good for large datasets and shows outliers better. ###
'''
Boxenplots (actually called letter-value plots in the original paper
and in the lvplot R package) show the distribution differently and
are better for bigger datasets. Classic boxplots can have too many
outliers and don't show as much information about the distribution.
Letter-value plots (boxenplots) start with the median (Q2, 50th percentile)
as the centerline. Each successive level outward contains half of the
remaining data. So the first two sections out from the centerline contain
50% of the data. After that, the next two sections contain 25% of the data.
This continues until we are at the outlier level. Each level out is shaded
lighter. There are 4 methods for calculating outliers (described in the
paper and available in seaborn). The default is to end up with around
5-8 outliers in each tail.
'''
for i in SelectFeatures:
    sns.boxenplot(y = i, data = df, palette = 'PRGn').set(title='boxenplot '+i)  #lightblue
    plt.show()
    


# PLOTTING ALL VARIABLES DISTRINBUTIONS ON THE SAME PLOT
# drag window to larger size to see


sns.boxplot(data = df, orient = 'h').set(title="boxplot all features")
# change to violinplot,
plt.show()

sns.violinplot(data = df, orient = 'h').set(title="violinplot all features")  # change to violinplot, 
plt.show()

##### BIVARIATE DISTRIBUTIONS

## SCATTERPLOT ##
for i in SelectFeatures:
    sns.scatterplot(x = df['AgeCategory'], y = df[i], hue = df['Outcome'])\
        .set(title='scatterplot '+i)  #hist = False for no histogram
    plt.show()



sns.scatterplot(x = df['BloodPressure'], y = df['BMI'], hue = df['Outcome'], style = df['PregCateg'], palette = 'deep')

plt.show()
sns.scatterplot(data = df, x = 'BloodPressure', y = 'BMI', hue = "Outcome",\
                size = "AgeCategory", sizes =(20,60), legend = "full", palette = 'deep')\
                .set(title='scatterplot '+i)  #hist = False for no histogram

plt.show()

# there's more in: https://seaborn.pydata.org/generated/seaborn.scatterplot.html

#### BARPLOTS

for i in SelectFeatures2:
        sns.barplot(x = 'AgeCategory', y = i, data = df, hue = 'Outcome', palette = 'PRGn')\
                .set(title='bivariate barplot '+'AgeCategory'+' and '+i)  #lightblue
        plt.show()



# BOXPLOTS - shows the minimum, maximum, 1st quartile and 3rd quartile.

for i in SelectFeatures2:
        sns.boxplot(x = 'AgeCategory', y = i, data = df, palette = 'PRGn')\
            .set(title='bivariate boxplot '+'AgeCategory'+' and '+i)  #lightblue
        plt.show()


# x axis variable is categorical and hue has only 2 values, then could do a split half and half violinplot.

sns.violinplot(x = 'AgeCategory', y = 'Insulin', data = df, hue = 'Outcome', \
                split = True, inner='quart', palette = 'PRGn')
plt.show()

# POINTPLOTS

for i in SelectFeatures2:
        sns.pointplot(x = 'AgeCategory', y = i, data = df, hue = 'Outcome', palette = 'PRGn')\
            .set(title='bivariate pointplot '+'AgeCategory'+' and '+i)  #lightblue
        plt.show()


### JOINTPLOT w/ scatterplot ####
'''
The joint plot is a way of understanding the relationship between
two variables and the distribution of individuals of each variable.
The joint plot mainly consists of three separate plots in which,
one of it was the middle figure that is used to see the relationship
between x and y. So, this area will give the information about the
joint distribution, while the remaining two areas will provide us
with the marginal distribution for the x-axis and y-axis.
'''
for i in SelectFeatures:
    for j in SelectFeatures:
        if i < j:
            sns.jointplot(data=df, x=i, y=j, hue="Outcome", kind = 'scatter') # hist, resid, kde
            plt.show()

        
# last one
fig, axes = plt.subplots(nrows=len(df.columns) // 2, ncols=2, figsize=(13, 10))


for idx, column in enumerate(df.drop(columns = 'Outcome')):
    row_idx = idx // 2 # int div
    col_idx = idx % 2 # remainder
    
    sns.kdeplot(df[df["Outcome"] == 1][column], alpha=0.5, fill=True, color="#000CEB", label="Diabetes", \
                ax=axes[row_idx, col_idx])
    sns.kdeplot(df[df["Outcome"] == 0][column], alpha=0.5, fill=True, color="#97B9F4", label="Normal", \
                ax=axes[row_idx, col_idx])  # alpha may be a transparency value, 0; transparent, 1 fully opaque
    
    axes[row_idx, col_idx].set_xlabel(column)
    axes[row_idx, col_idx].set_ylabel("Frequency")
    axes[row_idx, col_idx].set_title(f"{column} Distribution over Diabetes")
    axes[row_idx, col_idx].legend()

plt.tight_layout()
plt.show()



##### CORRELATIONS BETWEEN ATTRIBUTE PAIRS ###

'''
# Correlation matrices are an essential tool of exploratory data analysis.
# Correlation heatmaps contain the same information in a visually appealing way.
# What more: they show in a glance which variables are correlated, to what degree, in
# which direction, and alerts us to potential multicollinearity problems.
'''

###
Numerical = ['Pregnancies', 'PregCateg', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI', \
             'DPF', 'Age', 'AgeCategory','Outcome']

# DRAG out the table 
Corrmat = df[Numerical].corr()
###plt.figure(figsize=(10, 5), dpi=200)  # this makes the figure larger. but drops the numbers!!!
sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5).set(title="heatmap")
plt.show()

        








    









