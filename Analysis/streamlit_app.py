import streamlit as st
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import seaborn as sns
import adds
import matplotlib.pyplot as plt
import plotly.express as px


# Get important file paths, works when this script is run in any CWD
script_path=os.path.abspath(__file__)
repo_dir=os.path.dirname(script_path)


# Ensure script is run using Streamlit ---------------------------------------------------------------------------------
# If script wasn't run with Streamlit
if "--as-streamlit" not in sys.argv:
    # Rerun using Streamlit. Pass a sentinal "--as-streamlit" param to script,
    # which will be checked in the if statement
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path, "--", "--as-streamlit"],
                   cwd=repo_dir)

    # Don't execute the Streamlit code if not being run with Streamlit
    exit()

# Run Streamlit app ----------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="IDC Analysis Plotting Hub", page_icon="📊")
st.title("IDC Analysis Plotting Hub")


# SIDEBAR --------------------------------------------------------------------------------------------------------------
st.sidebar.title("Plotting Options")
with st.sidebar:

    # filter by solution type for image analysis plots
    st.header("Image Analysis Filters")
    option1=st.selectbox("Solution:", ["DI Water", "Adipic Acid - 1.24mM", "Adipic Acid - 0.712mM","Succinic 0.388mM",
                                      "Succinic 20mM","Succinic 1.425mM","Succinic 0.712 mM","Succinic 3.6mM",
                                      "Adipic Acid - 0.388mM"])

    # filter by pattern for failure time vs solution plots
    st.header("Failure Time vs Solution Filters")
    option2=st.selectbox("Pattern:",["1","4","7","10"])

    st.button("Apply Filters")
    st.header("Settings")
    if st.button("Update Cached Data"):
        st.write("Updating...")
        subprocess.run([sys.executable, "update_cache.py"], cwd=repo_dir)
        st.write("Done!")
# ----------------------------------------------------------------------------------------------------------------------

column1, column2=st.columns(2)

with column1:

    # RGB 3D Plot ----------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("3D RGB Analysis")
        st.text("Maps RGB values to XYZ coordinates to view the average color of all boards, "
                "separated by board type and pristine/exposed")
        def RGB_3D(option1):
            master=adds.get_master()
            master.dropna(subset="Pattern", inplace=True)

            # filter by solution choice
            master=master[master["Solution"]==option1]

            master["Pattern"]=master["Pattern"].apply(int).apply(str)

            # Melt the RGB columns
            master=master.melt(id_vars=["Pattern", "Board ID", "Sensor"],
                                 value_vars=["R_PRISTINE", "G_PRISTINE", "B_PRISTINE", "R_EXPOSED", "G_EXPOSED",
                                             "B_EXPOSED"],
                                 var_name="Channel_Age", value_name="Value")

            # Split "Channel_Age" into "Channel" and "Age"
            master[["Channel", "Age"]]=master["Channel_Age"].str.extract(r"([RGB])_(PRISTINE|EXPOSED)")

            # Pivot to get R, G, B in separate columns, with "Age" as one of the columns
            master=master.pivot_table(index=["Pattern", "Board ID", "Sensor", "Age"], columns="Channel",
                                        values="Value"
                                        ).reset_index()

            fig=px.scatter_3d(master, x="R", y="G", z="B", color="Pattern", symbol="Age",
                                symbol_map={"PRISTINE": "circle-open", "EXPOSED": "circle"}, opacity=0.6,
                                hover_data=["Pattern", "Board ID", "Sensor"])

            st.plotly_chart(fig)


        RGB_3D(option1)
    # ----------------------------------------------------------------------------------------------------------------------



    # RGB Boxplots ---------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("RGB Boxplots")
        st.text("Plots the differences in average RGB channels for pristine VS exposed boards")
        def RGB_boxplots(option1):
            master=adds.get_master()

            # filter by solution choice
            master=master[master["Solution"]==option1]

            # Add columns for RGB difference
            master["Red"]=master["R_EXPOSED"]-master["R_PRISTINE"]
            master["Green"]=master["G_EXPOSED"]-master["G_PRISTINE"]
            master["Blue"]=master["B_EXPOSED"]-master["B_PRISTINE"]

            # Convert to long for easy plotting
            # A channel column will be added, storing "R_Diff"...
            master=pd.melt(master, id_vars=["Board ID", "Sensor", "Pattern"], value_vars=["Red", "Green", "Blue"],
                             var_name="Channel", value_name="Channel Difference")

            # Create a FacetGrid
            g=sns.FacetGrid(data=master, col="Channel", margin_titles=True, hue="Channel",
                              palette={"Red": "#FF0000", "Green": "#00FF00", "Blue": "#0000FF"})

            # Create a lineplot on the FacetGrid
            g.map_dataframe(sns.boxplot, x="Pattern", y="Channel Difference", )

            # Set the text of the titles
            g.set_titles(col_template="{col_name}")

            # Set ticks to ints, not floats
            g.set_xticklabels([1, 4, 7, 10])

            st.pyplot(g)


        RGB_boxplots(option1)
    # ----------------------------------------------------------------------------------------------------------------------



    # Grayscale Boxplots ---------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Grayscale Boxplots")
        st.text("Plots the average brightness of each board")
        def grayscale(option1):
            master=adds.get_master()

            master=master[master["Solution"]==option1]

            # Add column for brightness difference
            master["Brightness Difference"]=master["Brightness Exposed"] - master["Brightness Pristine"]

            # make pattern categorical and ordered to make sure it is plotted correctly
            master["Pattern"]=pd.Categorical(master["Pattern"], categories=[1, 4, 7, 10], ordered=True)

            # plot
            fig, ax=plt.subplots(figsize=(10, 6))
            sns.boxplot(data=master, x="Pattern", y="Brightness Difference", ax=ax)
            st.pyplot(fig)


        grayscale(option1)
    # ----------------------------------------------------------------------------------------------------------------------

    # Failure Time vs Solution ---------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Failure Time vs Solution")
        st.text("Plots failure time as function of solution, separated by board type and sensor")
        def failure_time():

            # Get master data
            master=adds.get_master()

            # Drop NaN rows
            master.dropna(subset="Voltage", inplace=True)

            # Add column to store failure time in seconds
            master["Failure Time (s)"]=master["Time to Failure (ms)"] / 1000

            # Plot data for each unique voltage
            for voltage in master["Voltage"].unique():
                # Create a FacetGrid
                g=sns.FacetGrid(data=master[master["Voltage"] == voltage], row="Pattern", row_order=[1, 4, 7, 10],
                                  hue="Sensor",
                                  palette={"U1": "#FF0000", "U2": "#B6FF00", "U3": "#00FFFF", "U4": "#7F00FF"},
                                  margin_titles=True, sharex=False, sharey=False)

                # Create scatterplots on the FacetGrid
                g.map_dataframe(sns.pointplot, x="Solution", y="Failure Time (s)",
                                order=["DI Water", "Adipic Acid - 0.388mM", "Adipic Acid - 0.712mM",
                                       "Adipic Acid - 1.24mM",
                                       "Succinic 0.388mM", "Succinic 0.712 mM", "Succinic 1.425mM", "Succinic 3.6mM"],
                                errorbar=None)

                # Shrink font size
                g.tick_params(labelsize="small")

                # Set the text of the titles, which are already positioned properly
                g.set_titles(row_template="Pattern {row_name}", col_template="{col_name}")

                # Instead of an axis being L-shaped, make it a box
                for ax in g.axes.flat:
                    ax.spines["top"].set_visible(True)
                    ax.spines["right"].set_visible(True)

                # Add legend
                g.add_legend(title="Sensor", edgecolor="#000000", frameon=True)

                # Add main title
                g.figure.suptitle(f"Mean Failure Time Vs Solution, by Pattern, and Sensor ({int(voltage)}V)")

                # Adjust spacing
                g.figure.subplots_adjust(left=0.06, bottom=0.08, right=0.91, top=0.94)

                st.pyplot(g.figure)


        failure_time()
    # ----------------------------------------------------------------------------------------------------------------------



with column2:

    # Scatterplot Matrix ---------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Scatterplot Matrix")
        st.text("Plots a scatterplot matrix for all variable pairs")
        def scatter():
            # Get master data
            master=adds.get_master()

            master=master[
                ["Pattern", "Time to Failure (ms)", "Voltage", "pH", "Dendrite Score", "Brightness Pristine",
                 "Brightness Exposed", "Brightness Difference"]]

            # Drop NA values
            master.dropna(axis=1, how="all", inplace=True)

            # Drop non-numeric columns
            master=master.select_dtypes(include=["number"])

            # Plot
            axes=pd.plotting.scatter_matrix(master, figsize=(10, 10), alpha=1)

            for ax in axes.flatten():
                ax.xaxis.label.set_rotation(45)
                ax.xaxis.label.set_ha("right")
                ax.yaxis.label.set_rotation(45)
                ax.yaxis.label.set_ha("right")

            fig=axes[0, 0].get_figure()
            plt.tight_layout()

            st.pyplot(fig)


        scatter()
    # ----------------------------------------------------------------------------------------------------------------------



    # Correlation Heatmap --------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Correlation Heatmap")
        st.text("Plots the correlations between all variable pairs")
        def heatmap():
            # Get master
            master=adds.get_master()

            # edited
            master=master[
                ["Pattern", "Time to Failure (ms)", "Voltage", "pH", "Dendrite Score", "Brightness Pristine",
                 "Brightness Exposed", "Brightness Difference"]]

            # Drop columns that are entirely NaN
            master.dropna(axis=1, how="all", inplace=True)
            # Drop columns that are non-numeric
            master=master.select_dtypes(include=["number"])

            # Create a heatmap of the correlation matrix
            fig, ax=plt.subplots()
            sns.heatmap(master.corr(), annot=True, cmap="coolwarm", ax=ax)
            plt.tight_layout()
            st.pyplot(fig)


        heatmap()
    # ----------------------------------------------------------------------------------------------------------------------



    # pH plot --------------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("pH Plot")
        st.text("Plots failure time as a function of pH")
        def plot_ph():
            df=adds.get_master()

            # Remove solutions with no recorded Ph so they don't take up space in the legend
            df=df[(df["Solution"]=="Adipic Acid - 1.24mM")|(df["Solution"]=="Adipic Acid - 0.712mM")|(
                    df["Solution"]=="Adipic Acid - 0.388mM")|(df["Solution"]=="Succinic 0.388mM")]

            # Plot
            fig, ax=plt.subplots(figsize=(8, 6))
            sns.scatterplot(x="pH", y="Time to Failure (ms)", data=df, hue="Solution", ax=ax)

            ax.set_title("Time to Failure (ms) vs. pH by Solution Type")

            # Move legend to the right of the plot
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

            # Tight layout so the legend doesn't get cut off
            plt.tight_layout()
            st.pyplot(fig)


        plot_ph()

    # ----------------------------------------------------------------------------------------------------------------------

    # CF/CV Plots ----------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("CF and CV Plots")
        st.text("Plots CF and CV Data")
        def CF_CV():

            # get CF and CV data
            CF=adds.get_master_cf_or_cv(cf_or_cv="CF")
            CV=adds.get_master_cf_or_cv(cf_or_cv="CV")

            # filter out files with bad data
            CF=CF[(CF["Capacitance (F)"]>0)&(CF["Capacitance (F)"]<100)]
            CV=CV[(CV["Capacitance (F)"]>0)&(CV["Capacitance (F)"]<100)]

            # list of sensor names and colors for plotting
            sensors=["U1", "U2", "U3", "U4"]
            colors=["c", "m", "y", "#47E183"]

            # take the average across all data for each sensor and frequency for CF data
            if CF is not None and not CF.empty:
                CF_average=CF.groupby(["Sensor", "Frequency (Hz)"]).agg(
                    {"Capacitance (F)": "mean", "Impedance (O)": "mean",
                     "Phase Angle (D)": "mean"}).reset_index()

            # take the average across all data for each sensor and voltage for CV data
            if CV is not None and not CV.empty:
                CV_average=CV.groupby(["Sensor", "Voltage (V)"]).agg(
                    {"Capacitance (F)": "mean", "Impedance (O)": "mean",
                     "Phase Angle (D)": "mean"}).reset_index()

            # create figure and subplots
            fig, axes=plt.subplots(2, 3, figsize=(15, 8))

            # flatten array
            ax1, ax2, ax3, ax4, ax5, ax6=axes.flatten()

            # plot CF data - one line per sensor
            for i, sensor in enumerate(sensors):
                sensor_data=CF_average[CF_average["Sensor"]==sensor]

                # plot Capacitance (F) vs. Frequency (Hz)
                ax1.plot(sensor_data["Frequency (Hz)"], sensor_data["Capacitance (F)"], c=colors[i], label=sensor)

                # plot Impedance (O) vs. Frequency (Hz)
                ax2.plot(sensor_data["Frequency (Hz)"], sensor_data["Impedance (O)"], c=colors[i], label=sensor)

                # plot Phase Angle (D) vs. Frequency (Hz)
                ax3.plot(sensor_data["Frequency (Hz)"], sensor_data["Phase Angle (D)"], c=colors[i], label=sensor)

            ax1.set_title("Capacitance (F) vs. Frequency (Hz)")
            ax1.set_ylabel("Capacitance (F)")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylim(0, 25)
            ax1.legend()

            ax2.set_title("Impedance (O) vs. Frequency (Hz)")
            ax2.set_ylabel("Impedance (O)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylim(0, 10e6)
            ax2.legend()

            ax3.set_title("Phase Angle (D) vs. Frequency (Hz)")
            ax3.set_ylabel("Phase Angle (D)")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylim(0, -100)
            ax3.legend()

            # plot CV data - one line per sensor
            for i, sensor in enumerate(sensors):
                sensor_data=CV_average[CV_average["Sensor"] == sensor]

                # plot "Capacitance (F) vs. Voltage (V)
                ax4.plot(sensor_data["Voltage (V)"], sensor_data["Capacitance (F)"], c=colors[i], label=sensor)

                # plot Impedance (O) vs. Voltage (V)
                ax5.plot(sensor_data["Voltage (V)"], sensor_data["Impedance (O)"], c=colors[i], label=sensor)

                # plot Phase Angle (D) vs. Voltage (V)
                ax6.plot(sensor_data["Voltage (V)"], sensor_data["Phase Angle (D)"], c=colors[i], label=sensor)

            ax4.set_title("Capacitance (F) vs. Voltage (V)")
            ax4.set_ylabel("Capacitance (F)")
            ax4.set_xlabel("Voltage (V)")
            ax4.set_ylim(0, 25)
            ax4.legend()

            ax5.set_title("Impedance (O) vs. Voltage (V)")
            ax5.set_ylabel("Impedance (O)")
            ax5.set_xlabel("Voltage (V)")
            ax5.set_ylim(0, 10e6)
            ax5.legend()

            ax6.set_title("Phase Angle (D) vs. Voltage (V)")
            ax6.set_ylabel("Phase Angle (D)")
            ax6.set_xlabel("Voltage (V)")
            ax6.set_ylim(0, -100)
            ax6.legend()

            plt.tight_layout()
            st.pyplot(fig)

        CF_CV()
    # ----------------------------------------------------------------------------------------------------------------------



    # Current vs Time ------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Current vs Time")
        st.text("Plots current as a function of time for each tested sensor separated by solution and board type")
        def current_vs_time():

            # Get joined data
            master_current_time=adds.get_master_current_time()

            # Add a unique sensor identifier
            master_current_time["Sensor ID"]=master_current_time["Board ID"] + "_" + master_current_time["Sensor"]

            # Plot data for each unique voltage
            for voltage in master_current_time["Voltage"].unique():
                # Create a FacetGrid
                g=sns.FacetGrid(data=master_current_time[master_current_time["Voltage"]==voltage], row="Pattern",
                                  row_order=[1, 4, 7, 10], col="Solution",
                                  col_order=["DI Water", "Adipic Acid - 0.388mM", "Adipic Acid - 0.712mM",
                                             "Adipic Acid - 1.24mM", "Succinic 0.388mM", "Succinic 0.712 mM",
                                             "Succinic 1.425mM",
                                             "Succinic 3.6mM"], hue="Sensor",
                                  palette={"U1": "#FF0000", "U2": "#B6FF00", "U3": "#00FFFF",
                                           "U4": "#7F00FF"}, margin_titles=True, sharex=False, sharey=False)

                # Create a lineplot on the FacetGrid
                g.map_dataframe(sns.lineplot, x="Time (ms)", y="Current (mA)", units="Sensor ID", estimator=None)

                # Set the text of the titles, which are already positioned properly
                g.set_titles(row_template="Pattern {row_name}", col_template="{col_name}")

                # Remove all ticks and tick labels
                g.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])

                # Instead of an axis being L-shaped, make it a box
                for ax in g.axes.flat:
                    ax.spines["top"].set_visible(True)
                    ax.spines["right"].set_visible(True)

                # Adjust spacing
                g.figure.subplots_adjust(wspace=0, hspace=0, left=0.03, bottom=0.05, right=0.97, top=0.92)

                # Add legend
                g.add_legend(title="Sensor", edgecolor="#000000", frameon=True)

                # Add main title
                g.figure.suptitle(f"Current Vs Time, by Solution, Pattern, and Sensor ({int(voltage)}V)")

                st.pyplot(g.figure)


        current_vs_time()
    # ----------------------------------------------------------------------------------------------------------------------


# option to view master data frame
data=pd.read_csv("../IDCSubmersionMasterlist_20250505.csv")
with st.expander("See Data"):
    st.dataframe(data)
