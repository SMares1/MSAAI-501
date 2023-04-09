import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

##Data Preparation
dataset = pd.read_csv('bank-full.csv', engine = 'python')
print(dataset.head())
print(dataset.columns)
dataset.describe()

new_column_names = dataset.columns

dataset.columns = new_column_names.str.replace('.','_')

##Dataset Visualization
print(dataset['y'].value_counts())
dataset['y'].hist()
print(plt.show())
#percentage of those who said (y/n) with target variable
print(dataset['y'].value_counts(normalize=True, dropna=False))

#plotting the target variable to determine how imbalanced it is
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,6))

sns.countplot(data = dataset, x = 'y')

plt.xlabel('Decision (Y/N)')
plt.title('Has The Client Subscribed a Term Deposit?')
plt.tight_layout();
plt.show()

#graph of age distribution
plt.figure(figsize = (10, 12))
plt.style.use('default')
age_graph = sns.displot(data=dataset, x='age', hue='y', bins=30, kde = False, legend=False)
plt.title('Age Distribution of Consumers')
plt.xlabel('Age')
plt.show()

##age distribution graph based on decision
#age distribution based on 'yes' decision
fig, axes = plt.subplots(1,2)
plt.style.use('default')
sns.set(rc={"figure.figsize":(8, 4)})
sns.histplot(dataset.loc[dataset['y']=='yes']['age'], bins=30, kde = True, color='blue' , ax=axes[0])
axes[0].set_xlabel("Age", fontsize = 8)
axes[0].set_ylabel('count')
axes[0].set_title('Age Distribution if (Yes)')

#age distribution based on 'no' decision
sns.histplot(dataset.loc[dataset['y']=='no']['age'], bins=30, kde = True, color='red', ax=axes[1])
axes[1].set_xlabel("Age", fontsize = 10)
axes[1].set_ylabel('count')
axes[1].set_title('Age Distribution if (No)')
print(plt.show())

#job distribution graphs with age
graph_layout = sns.FacetGrid(dataset, col = 'job', hue = 'y', col_wrap=3)
graph_layout.map(plt.hist, 'age')
print(plt.show())

#education distribution graphs with age
graph_layout = sns.FacetGrid(dataset, col = 'education', hue = 'y', col_wrap=4)
graph_layout.map(plt.hist, 'age')
print(plt.show())

#marital distribution graphs with age
graph_layout = sns.FacetGrid(dataset, col = 'marital', hue = 'y', col_wrap=4)
graph_layout.map(plt.hist, 'age')
print(plt.show())

#duration of the last phone call based on decision (Y/N)
sns.catplot(data = dataset, x = 'y', y = 'duration')
print(plt.show())

##Categorical Pre-Processing

dataset['job'] = dataset['job'].str.replace('-','_').str.replace('.','', regex=True)
print(dataset['job'].value_counts())

#print out 'poutcome', the outcome of the previous marketing campaign
print(dataset['poutcome'].value_counts())

contingency_table = pd.crosstab(dataset['y'], dataset['job'], margins = True)
print(contingency_table)