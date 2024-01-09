import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

###################################################################
#                             GEMVER                              #
###################################################################

#Path to the directory containing the CSV files
directory = "./results/gemver"

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
baseline_name = 'baseline'
base_times = df[df['method'] == baseline_name].set_index('N')['time [s]'].groupby('N').mean()
print("Baseline mean times:\n", base_times)

# Step 2: Merge the base times with the original DataFrame
df = df.merge(base_times, left_on='N', right_index=True, suffixes=('', '_base'))

df['speedup'] = df['time [s]_base'] / df['time [s]']
df.drop(['time [s]_base'], axis=1, inplace=True)
df = df[df['method'] != baseline_name]


sns.set_theme()
plot = sns.lineplot(
    data=df, x="N", y="speedup", hue="method", errorbar="sd", estimator=np.median
).set(title='Gemver Speedups')
# plt.xscale('log')

file_path = "./results/plot_gemver_openmp.png"
plt.savefig(file_path)
plt.show()

###################################################################
#                           TRISOLV                               #
###################################################################

#Path to the directory containing the CSV files
directory = "./results/trisolv"

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
baseline_name = 'baseline'
base_times = df[df['method'] == baseline_name].set_index('N')['time [s]'].groupby('N').mean()
print("Baseline mean times:\n", base_times)

# Step 2: Merge the base times with the original DataFrame
df = df.merge(base_times, left_on='N', right_index=True, suffixes=('', '_base'))

df['speedup'] = df['time [s]_base'] / df['time [s]']
df.drop(['time [s]_base'], axis=1, inplace=True)
df = df[df['method'] != baseline_name]

sns.set_theme()
sns.lineplot(
    data=df, x="N", y="speedup", hue="method", errorbar="sd", estimator=np.median
).set(title='Trisolv Speedups')

# plt.xscale('log')

file_path = "./results/plot_trisolv_openmp.png"
plt.savefig(file_path)
plt.show()
