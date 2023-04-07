import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

##Data Preparation
dataset = pd.read_csv('bank-full.csv', engine = 'python', delim_whitespace=True)
print(dataset.head())
print(dataset.shape)
print(dataset.columns)
print(dataset.describe())

label_encoder = LabelEncoder()
dataset.job = label_encoder.fit_transform(dataset.job)
dataset.marital = label_encoder.fit_transform(dataset.marital)
dataset.education = label_encoder.fit_transform(dataset.education)
dataset.default = label_encoder.fit_transform(dataset.default)