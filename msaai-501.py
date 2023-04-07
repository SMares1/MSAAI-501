import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('bank.csv', sep = "\n ", engine = 'python')
print(dataset.head())