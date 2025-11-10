import adds
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Required to show plt plots in Streamlit
matplotlib.use("TkAgg")

# Get master
master = adds.get_master()
# Drop columns that are entirely NaN
master.dropna(axis=1, how="all", inplace=True)
# Drop columns that are non-numeric
master = master.select_dtypes(include=["number"])

# Create a heatmap of the correlation matrix
sns.heatmap(master.corr(), annot=True, cmap="coolwarm")
plt.show()
