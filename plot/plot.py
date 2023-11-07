import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#Path to the directory containing the CSV files
directory = "./results"

# Initialize an empty list to store DataFrames
dataframes = []

# Iterate over all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path, sep=";")
        dataframes.append(df)

# Concatenate all DataFrames into one
df = pd.concat(dataframes, ignore_index=True)

sns.set_theme()
sns.lineplot(
    data=df, x="N", y="time [s]", hue="method", errorbar="sd", estimator=np.median
)
# plt.xscale('log')

plt.show()
file_path = "./results/plot.png"
plt.savefig(file_path)
