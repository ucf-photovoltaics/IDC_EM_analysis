# Plot current vs time from all the CurrentTime files. The plots are grouped by
# solution, pattern, sensor, and voltage. This plot uses the master_current_time
# joined data from cleans.py.

import adds
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Required to show plt plots in Streamlit
matplotlib.use("TkAgg")

# Get joined data
master_current_time = adds.get_master_current_time()

# Add a unique sensor identifier
master_current_time["Sensor ID"] = master_current_time["Board ID"] + "_" + master_current_time["Sensor"]

# Plot Data --------------------------------------------------------------------
# Plot for each unique voltage
for voltage in master_current_time["Voltage"].unique():
    # Create a FacetGrid
    g = sns.FacetGrid(
        data=master_current_time[master_current_time["Voltage"] == voltage],
        row="Pattern", row_order=[1, 4, 7, 10],
        col="Solution", col_order=["DI Water", "Adipic Acid - 0.388mM", "Adipic Acid - 0.712mM", "Adipic Acid - 1.24mM", "Succinic 0.388mM", "Succinic 0.712 mM", "Succinic 1.425mM", "Succinic 3.6mM"],
        hue="Sensor", palette={"U1":"#FF0000", "U2":"#B6FF00", "U3":"#00FFFF", "U4":"#7F00FF"},
        margin_titles=True,
        sharex=False, sharey=False
    )

    # Create a lineplot on the FacetGrid
    g.map_dataframe(
        sns.lineplot,
        x="Time (ms)", y="Current (mA)",
        units="Sensor ID", estimator=None
    )

    # Set the text of the titles, which are already positioned properly
    g.set_titles(
        row_template="Pattern {row_name}",
        col_template="{col_name}"
    )

    # Remove all ticks and tick labels
    g.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])

    # Instead of an axis being L-shaped, make it a box
    for ax in g.axes.flat:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    # Adjust spacing
    g.figure.subplots_adjust(
        wspace=0,
        hspace=0,
        left=0.03,
        bottom=0.05,
        right=0.97,
        top=0.92
    )

    # Add legend
    g.add_legend(title="Sensor", edgecolor="#000000", frameon=True)

    # Add main title
    g.figure.suptitle(f"Current Vs Time, by Solution, Pattern, and Sensor ({int(voltage)}V)")

    plt.show()
