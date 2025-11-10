# Boxplots that show brightness difference between pristine/exposed images for each board pattern
import adds
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
# Required to show plt plots in Streamlit
matplotlib.use("TkAgg")

master=adds.get_master()

# Add column for brightness difference
master["Brightness Difference"]=master["Brightness Exposed"]-master["Brightness Pristine"]

# make pattern categorical and ordered to make sure it is plotted correctly
master["Pattern"]=pd.Categorical(master["Pattern"], categories=[1, 4, 7, 10], ordered=True)

# plot
ax=sns.boxplot(data=master, x="Pattern", y="Brightness Difference")
plt.show()
