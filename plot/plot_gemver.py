import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from get_flops import get_flops_gemver

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

# Add flop count
df["flops"] = df.apply(lambda x: get_flops_gemver(x['method'], x['N']), axis=1)

# compute performance
df["Performance [Gflop/s]"] = df["flops"]/df["time [s]"]/10**9



# plot timing
sns.set_theme()
sns.lineplot(
    data=df, x="N", y="time [s]", hue="method", errorbar="sd", estimator=np.median, marker='o'
)

file_path = "./results/plot_gemver_timing.png"
plt.savefig(file_path)
plt.show()


# plot performance
sns.lineplot(
    data=df, x="N", y="Performance [Gflop/s]", hue="method", errorbar="sd", estimator=np.median, marker='o'
)

file_path = "./results/plot_gemver_performance.png"
plt.savefig(file_path)
plt.show()