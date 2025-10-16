# plots a scatterplot matrix of all variables to see how they are correlated
import pandas
import adds
import matplotlib
import matplotlib.pyplot as plt

# Required to show plt plots in Streamlit
matplotlib.use("TkAgg")

# Get master data
master=adds.get_master()

# Rename Ph column to be correct
master.rename(columns={'Ph': 'pH'}, inplace=True)

# Drop NA values
master.dropna(axis=1, how="all", inplace=True)

# Drop non-numeric columns
master=master.select_dtypes(include=["number"])

# Plot
axes=pandas.plotting.scatter_matrix(master, figsize=(10, 10), alpha=1)

# Rotate x and y labels so that they are more readable
[ax.xaxis.label.set(rotation=45, ha="right") for ax in axes.flatten()]
[ax.yaxis.label.set(rotation=45, ha="right") for ax in axes.flatten()]

plt.show()
