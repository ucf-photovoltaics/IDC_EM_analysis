# plots pH as a function of Time to Failure (ms), colored by solution type
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import adds

# Required to show plt plots in Streamlit
matplotlib.use("TkAgg")

def plot_ph():
    
    df=adds.get_master()

    # Remove solutions with no recorded Ph so they don't take up space in the legend
    df=df[(df["Solution"]=="Adipic Acid - 1.24mM")|(df["Solution"]=="Adipic Acid - 0.712mM")|(df["Solution"]=="Adipic Acid - 0.388mM")|(df["Solution"]=="Succinic 0.388mM")]

    # Rename Ph column to be correct
    df.rename(columns={'Ph': 'pH'}, inplace=True)
    # Plot
    ax=sns.scatterplot(x="pH", y="Time to Failure (ms)", data=df, hue="Solution")
    
    ax.set_title("Time to Failure (ms) vs. pH by Solution Type")

    # Move legend to the right of the plot
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Tight layout so the legend doesn't get cut off
    plt.tight_layout()
    plt.show()
    
plot_ph()
