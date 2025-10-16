# Plot mean failure time vs solution, separated by pattern, sensor, and voltage.

import adds
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Required to show plt plots in Streamlit
matplotlib.use("TkAgg")

# Get master data
master = adds.get_master()

# Drop NaN rows
master.dropna(subset="Voltage", inplace=True)

# Add column to store failure time in seconds
master["Failure Time (s)"] = master["Time to Failure (ms)"] / 1000

# Plot Data --------------------------------------------------------------------
# Plot for each unique voltage
for voltage in master["Voltage"].unique():
    # Create a FacetGrid
    g = sns.FacetGrid(
        data=master[master["Voltage"] == voltage],
        row="Pattern", row_order=[1, 4, 7, 10],
        hue="Sensor", palette={"U1":"#FF0000", "U2":"#B6FF00", "U3":"#00FFFF", "U4":"#7F00FF"},
        margin_titles=True,
        sharex=False, sharey=False
    )

    # Create scatterplots on the FacetGrid
    g.map_dataframe(
        sns.pointplot,
        x="Solution", y="Failure Time (s)",
        order=["DI Water", "Adipic Acid - 0.388mM", "Adipic Acid - 0.712mM", "Adipic Acid - 1.24mM", "Succinic 0.388mM", "Succinic 0.712 mM", "Succinic 1.425mM", "Succinic 3.6mM"],
        errorbar=None
    )

    # Shrink font size
    g.tick_params(labelsize="small")

    # Set the text of the titles, which are already positioned properly
    g.set_titles(
        row_template="Pattern {row_name}",
        col_template="{col_name}"
    )

    # Instead of an axis being L-shaped, make it a box
    for ax in g.axes.flat:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    # Add legend
    g.add_legend(title="Sensor", edgecolor="#000000", frameon=True)

    # Add main title
    g.figure.suptitle(f"Mean Failure Time Vs Solution, by Pattern, and Sensor ({int(voltage)}V)")

    # Adjust spacing
    g.figure.subplots_adjust(
        left=0.06,
        bottom=0.08,
        right=0.91,
        top=0.94
    )

    plt.show()
