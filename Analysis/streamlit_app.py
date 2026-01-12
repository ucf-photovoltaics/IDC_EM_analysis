import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn
import sklearn.model_selection as ms
import streamlit as st
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

import adds

# Get important file paths, works when this script is run in any CWD
script_path = os.path.abspath(__file__)
repo_dir = os.path.dirname(script_path)

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
    # filter by solution type
    st.header("Solution Filters")
    option1 = st.selectbox("Solution:", ["All", "DI Water", "Adipic Acid - 1.24mM", "Adipic Acid - 0.712mM",
                                         "Succinic 0.388mM", "Succinic 20mM", "Succinic 1.425mM", "Succinic 0.712 mM",
                                         "Succinic 3.6mM", "Adipic Acid - 0.388mM"])

    st.button("Apply Filters")
    st.header("Settings")
    if st.button("Update Cached Data"):
        st.write("Updating...")
        subprocess.run([sys.executable, "update_cache.py"], cwd=repo_dir)
        st.write("Done!")
# ----------------------------------------------------------------------------------------------------------------------

column1, column2 = st.columns(2)

with column1:
    # Random Forest ----------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Random Forest")
        st.text("")


        def random_forest():
            # Get master data
            master = adds.get_master()

            master["Brightness Difference"] = master["Brightness Exposed"] - master["Brightness Pristine"]
            master["Failure Time (s)"] = master["Time to Failure (ms)"] / 1000

            features = ["Board ID", "Pattern", "Solution", "Sensor", "Failure Time (s)", "Current", "Voltage",
                        "Brightness Difference"]

            # create a subset of the master data frame only with necessary features
            master_subset = master[features].copy()
            master_subset.dropna(inplace=True)

            # create label encoders for categorical variables
            le_pattern = LabelEncoder()
            le_solution = LabelEncoder()
            le_sensor = LabelEncoder()

            # apply label encoders, save true labels for plotting
            master_subset["True_Pattern"] = master_subset["Pattern"]
            master_subset["Pattern"] = le_pattern.fit_transform(master_subset["Pattern"])

            master_subset["True_Solution"] = master_subset["Solution"]
            master_subset["Solution"] = le_solution.fit_transform(master_subset["Solution"])

            master_subset["True_Sensor"] = master_subset["Sensor"]
            master_subset["Sensor"] = le_sensor.fit_transform(master_subset["Sensor"])

            # split data into X and Y
            X = master_subset[["Pattern", "Solution", "Sensor", "Failure Time (s)", "Voltage"]].copy()
            Y = master_subset[["Brightness Difference"]]

            # split into training and testing
            XTrain, XTest, YTrain, YTest = ms.train_test_split(X, Y, test_size=0.3, random_state=1)

            model = sklearn.ensemble.RandomForestRegressor(n_estimators=50, criterion="squared_error", random_state=1)

            # testing ranges
            depth_val = np.arange(2, 11)
            leaf_val = np.arange(1, 31, step=9)

            grid_s = [{"max_depth": depth_val, "min_samples_leaf": leaf_val}]

            # use GridSearchCV for cross validation and get the best depth/minimum samples
            cv = GridSearchCV(estimator=model, param_grid=grid_s, cv=ms.KFold(n_splits=10))

            # fit model
            cv.fit(XTrain, YTrain.values.ravel())
            best_depth = cv.best_params_["max_depth"]
            best_min_samples = cv.best_params_["min_samples_leaf"]

            new_model = sklearn.ensemble.RandomForestRegressor(n_estimators=50, criterion="squared_error",
                                                               max_depth=best_depth, min_samples_leaf=best_min_samples,
                                                               random_state=1)

            # fit the model
            model_fitted = new_model.fit(XTrain, YTrain.values.ravel())

            # get predictions
            predictions = model_fitted.predict(XTest)

            # model performance metrics
            r2 = r2_score(YTest, predictions)
            mse = mean_squared_error(YTest, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(YTest, predictions)

            print("\nRandom Forest Performance Metrics:\n")
            print(f"R2 Score: {r2}", f"MSE: {mse}")
            print(f"RMSE: {rmse}", f"MAE: {mae}")

            results = pd.DataFrame({"Actual": YTest.values.ravel(), "Predicted": predictions})

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results["Actual"], results["Predicted"], alpha=0.8)
            ax.plot([results["Actual"].min(), results["Actual"].max()], [results["Actual"].min(),
                                                                         results["Actual"].max()], "--", c="#023858")
            ax.annotate("r-squared = {:.3f}".format(r2), (-25, 32))
            ax.set_xlabel("Actual Dendrite Growth")
            ax.set_ylabel("Predicted Dendrite Growth")
            ax.set_title("Random Forest: Actual vs Predicted")
            st.pyplot(fig)


        random_forest()
    # ------------------------------------------------------------------------------------------------------------------

    # XGBoost ----------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("XGBoost")
        st.text("Dendrite Growth/Failure Time (s)/Capacitance")


        def xgboost():
            # Get master data
            master = adds.get_master()

            master["Brightness Difference"] = master["Brightness Exposed"] - master["Brightness Pristine"]
            master["Failure Time (s)"] = master["Time to Failure (ms)"] / 1000

            features = ["Board ID", "Pattern", "Solution", "Sensor", "Failure Time (s)", "Current", "Voltage",
                        "Brightness Difference"]

            # create a subset of the master data frame only with necessary features
            master_subset = master[features].copy()
            master_subset.dropna(inplace=True)

            # create label encoders for categorical variables
            le_pattern = LabelEncoder()
            le_solution = LabelEncoder()
            le_sensor = LabelEncoder()

            # apply label encoders
            master_subset["Pattern"] = le_pattern.fit_transform(master_subset["Pattern"])

            master_subset["Solution"] = le_solution.fit_transform(master_subset["Solution"])

            master_subset["Sensor"] = le_sensor.fit_transform(master_subset["Sensor"])

            # split data into X and Y
            X = master_subset[["Pattern", "Solution", "Sensor", "Failure Time (s)", "Voltage"]].copy()
            y = master_subset[["Brightness Difference"]]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            param_grid = {"max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1, 0.2], "subsample": [0.8, 1.0],
                          "colsample_bytree": [0.8, 1.0]}

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred_tuned = best_model.predict(X_test)

            # model performance metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            print("\nXGBoost Performance Metrics:\n")
            print(f"R2 Score: {r2}", f"MSE: {mse}")
            print(f"RMSE: {rmse}", f"MAE: {mae}")

            results = pd.DataFrame({"Actual": y_test.values.ravel(), "Predicted": y_pred_tuned})

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results["Actual"], results["Predicted"], alpha=0.8)
            ax.plot([results["Actual"].min(), results["Actual"].max()], [results["Actual"].min(),
                                                                         results["Actual"].max()], "--", c="#023858")
            ax.annotate("r-squared = {:.3f}".format(r2), (-30, 25))
            ax.set_xlabel("Actual Dendrite Growth")
            ax.set_ylabel("Predicted Dendrite Growth")
            ax.set_title("XGBoost: Actual vs Predicted")
            st.pyplot(fig)

            # since xgboost is performing better, plot feature importance
            importance = model.get_booster().get_score(importance_type="weight")

            feature_importance = pd.DataFrame(
                {"Feature": list(importance.keys()), "Importance": list(importance.values())
                 }).sort_values(by="Importance", ascending=False)

            top_features = feature_importance.head().sort_values("Importance")
            fig = px.bar(top_features, x="Importance", y="Feature", orientation="h",
                         title=f"XGBoost Feature Importance",
                         labels={"Importance": "Importance"}, color="Importance", color_continuous_scale="PuBu")

            fig.update_layout(
                height=600,
                showlegend=False,
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                title_font=dict(color="black"),
                xaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                yaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="black"),
                    title_font=dict(color="black")
                )
            )

            st.plotly_chart(fig, use_container_width=True)


            # for failure time

            # split data into X and Y
            X2 = master_subset[["Pattern", "Solution", "Sensor", "Brightness Difference", "Voltage"]].copy()
            y2 = master_subset[["Failure Time (s)"]]

            X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

            model2 = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
            model2.fit(X_train2, y_train2)
            y_pred2 = model2.predict(X_test2)

            param_grid = {"max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1, 0.2], "subsample": [0.8, 1.0],
                          "colsample_bytree": [0.8, 1.0]}

            grid_search = GridSearchCV(estimator=model2, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

            grid_search.fit(X_train2, y_train2)

            best_model2 = grid_search.best_estimator_
            y_pred_tuned2 = best_model2.predict(X_test2)

            # model performance metrics
            r2 = r2_score(y_test2, y_pred2)
            mse = mean_squared_error(y_test2, y_pred2)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test2, y_pred2)

            print("\nXGBoost Performance Metrics:\n")
            print(f"R2 Score: {r2}", f"MSE: {mse}")
            print(f"RMSE: {rmse}", f"MAE: {mae}")

            results = pd.DataFrame({"Actual": y_test2.values.ravel(), "Predicted": y_pred_tuned2})

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results["Actual"], results["Predicted"], alpha=0.8)
            ax.plot([results["Actual"].min(), results["Actual"].max()], [results["Actual"].min(),
                                                                         results["Actual"].max()], "--", c="#006548")
            ax.annotate("r-squared = {:.3f}".format(r2), (-30, 25))
            ax.set_xlabel("Actual Failure Time (s)")
            ax.set_ylabel("Predicted Failure Time (s)")
            ax.set_title("XGBoost: Actual vs Predicted")
            st.pyplot(fig)

            # since xgboost is performing better, plot feature importance
            importance2 = model2.get_booster().get_score(importance_type="weight")

            feature_importance = pd.DataFrame(
                {"Feature": list(importance2.keys()), "Importance": list(importance2.values())
                 }).sort_values(by="Importance", ascending=False)

            top_features = feature_importance.head().sort_values("Importance")
            fig = px.bar(top_features, x="Importance", y="Feature", orientation="h",
                         title=f"XGBoost Feature Importance",
                         labels={"Importance": "Importance"}, color="Importance", color_continuous_scale="Greens")

            fig.update_layout(
                height=600,
                showlegend=False,
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                title_font=dict(color="black"),
                xaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                yaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="black"),
                    title_font=dict(color="black")
                )
            )

            st.plotly_chart(fig, use_container_width=True)


        xgboost()

    # ------------------------------------------------------------------------------------------------------------------

    # RGB 3D Plot ------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("3D RGB Analysis")
        st.text("Maps RGB values to XYZ coordinates to view the average color of all boards, "
                "separated by board type and pristine/exposed")


        def RGB_3D(option1):
            master = adds.get_master()
            master.dropna(subset="Pattern", inplace=True)

            # filter by solution choice
            if option1 != "All":
                master = master[master["Solution"] == option1]

            master["Pattern"] = master["Pattern"].apply(int).apply(str)

            # Melt the RGB columns
            master = master.melt(id_vars=["Pattern", "Board ID", "Sensor"], value_vars=["R_PRISTINE", "G_PRISTINE",
                                                                                        "B_PRISTINE", "R_EXPOSED",
                                                                                        "G_EXPOSED", "B_EXPOSED"],
                                 var_name="Channel_Age", value_name="Value")

            # Split "Channel_Age" into "Channel" and "Age"
            master[["Channel", "Age"]] = master["Channel_Age"].str.extract(r"([RGB])_(PRISTINE|EXPOSED)")

            # Pivot to get R, G, B in separate columns, with "Age" as one of the columns
            master = master.pivot_table(index=["Pattern", "Board ID", "Sensor", "Age"], columns="Channel",
                                        values="Value"
                                        ).reset_index()

            fig = px.scatter_3d(master, x="R", y="G", z="B", color="Pattern", symbol="Age", symbol_map={"PRISTINE":
                                                                                                            "circle-open",
                                                                                                        "EXPOSED": "circle"},
                                opacity=0.6, hover_data=["Pattern", "Board ID", "Sensor"],
                                width=800, height=600)

            st.plotly_chart(fig)


        RGB_3D(option1)
    # ------------------------------------------------------------------------------------------------------------------

    # RGB Boxplots -----------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("RGB Boxplots")
        st.text("Plots the differences in average RGB channels for pristine VS exposed boards")


        def RGB_boxplots(option1):
            master = adds.get_master()

            # filter by solution choice
            if option1 != "All":
                master = master[master["Solution"] == option1]

            # Add columns for RGB difference
            master["Red"] = master["R_EXPOSED"] - master["R_PRISTINE"]
            master["Green"] = master["G_EXPOSED"] - master["G_PRISTINE"]
            master["Blue"] = master["B_EXPOSED"] - master["B_PRISTINE"]

            master["Pattern"] = pd.Categorical(master["Pattern"], categories=[1, 4, 7, 10], ordered=True)

            # Convert to long for easy plotting
            # A channel column will be added, storing "R_Diff"...
            master = pd.melt(master, id_vars=["Board ID", "Sensor", "Pattern"], value_vars=["Red", "Green", "Blue"],
                             var_name="Channel", value_name="Channel Difference")

            # Create a FacetGrid
            g = sns.FacetGrid(data=master, col="Channel", margin_titles=True, hue="Channel", palette={"Red": "#FF0000",
                                                                                                      "Green": "#00FF00",
                                                                                                      "Blue": "#0000FF"})

            # Create a lineplot on the FacetGrid
            g.map_dataframe(sns.boxplot, x="Pattern", y="Channel Difference", )

            # Set the text of the titles
            g.set_titles(col_template="{col_name}")

            st.pyplot(g)


        RGB_boxplots(option1)
    # ------------------------------------------------------------------------------------------------------------------

    # Grayscale Boxplots -----------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Grayscale Boxplots")
        st.text("Plots the average brightness of each board - indicates dendrite growth")


        def grayscale(option1):
            master = adds.get_master()

            # filter by solution choice
            if option1 != "All":
                master = master[master["Solution"] == option1]

            # Add column for brightness difference
            master["Brightness Difference"] = master["Brightness Exposed"] - master["Brightness Pristine"]

            # make pattern categorical and ordered to make sure it is plotted correctly
            master["Pattern"] = pd.Categorical(master["Pattern"], categories=[1, 4, 7, 10], ordered=True)

            # plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=master, x="Pattern", y="Brightness Difference", hue="Sensor", palette="Spectral", ax=ax)

            # change title based on solution choice
            if option1 == "All":
                ax.set_title("Brightness Difference - All Solutions")
            else:
                ax.set_title(f"Brightness Difference - {option1}")

            st.pyplot(fig)


        grayscale(option1)
    # ------------------------------------------------------------------------------------------------------------------

    # Failure Time vs Solution -----------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Failure Time vs Solution")
        st.text("Plots failure time as function of solution, separated by board type and sensor")


        def failure_time(option1):

            # get master data
            master = adds.get_master()

            # drop rows with missing values
            master.dropna(subset="Voltage", inplace=True)

            # filter by solution choice
            if option1 != "All":
                master = master[master["Solution"] == option1]

            # add column to store failure time values
            master["Failure Time (s)"] = master["Time to Failure (ms)"] / 1000

            # log transformation of failure time for readability
            master["Log(Failure Time (s))"] = np.log10(master["Failure Time (s)"] + 1)

            master["Pattern"] = pd.Categorical(master["Pattern"], categories=[1, 4, 7, 10], ordered=True)

            # plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=master, x="Pattern", y="Log(Failure Time (s))", hue="Sensor", ax=ax, palette="Spectral")

            # change title based on solution choice
            if option1 == "All":
                ax.set_title("Failure Time - All Solutions")
            else:
                ax.set_title(f"Failure Time - {option1}")

            st.pyplot(fig)


        failure_time(option1)
    # ------------------------------------------------------------------------------------------------------------------

    # Scatterplot Matrix --------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Scatterplot Matrix")
        st.text("Plots a scatterplot matrix for all variable pairs")


        def scatter():
            # Get master data
            master = adds.get_master()

            master["Brightness Difference"] = master["Brightness Exposed"] - master["Brightness Pristine"]

            master = master[
                ["Pattern", "Time to Failure (ms)", "Voltage", "pH", "Dendrite Score", "Brightness Pristine",
                 "Brightness Exposed", "Brightness Difference"]]

            # Drop NA values
            master.dropna(axis=1, how="all", inplace=True)

            # Drop non-numeric columns
            master = master.select_dtypes(include=["number"])

            # Plot
            axes = pd.plotting.scatter_matrix(master, figsize=(10, 10), alpha=1)

            for ax in axes.flatten():
                ax.xaxis.label.set_rotation(45)
                ax.xaxis.label.set_ha("right")
                ax.yaxis.label.set_rotation(45)
                ax.yaxis.label.set_ha("right")

            fig = axes[0, 0].get_figure()
            plt.tight_layout()

            st.pyplot(fig)


        scatter()
    # ------------------------------------------------------------------------------------------------------------------

    # Correlation Heatmap ----------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Correlation Heatmap")
        st.text("Plots the correlations between all variable pairs")


        def heatmap():
            # Get master
            master = adds.get_master()
            master["Brightness Difference"] = master["Brightness Exposed"] - master["Brightness Pristine"]

            master = master[
                ["Pattern", "Time to Failure (ms)", "Voltage", "pH", "Dendrite Score", "Brightness Pristine",
                 "Brightness Exposed", "Brightness Difference"]]

            # Drop columns that are entirely NaN
            master.dropna(axis=1, how="all", inplace=True)
            # Drop columns that are non-numeric
            master = master.select_dtypes(include=["number"])

            # Create a heatmap of the correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(master.corr(), annot=True, cmap="coolwarm", ax=ax)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va="center")

            plt.tight_layout()
            st.pyplot(fig)


        heatmap()
    # ------------------------------------------------------------------------------------------------------------------

with column2:
    # Cluster Analysis -------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Cluster Analysis")


        def cluster_analysis():

            # Get master data
            master = adds.get_master()

            master["Brightness Difference"] = master["Brightness Exposed"] - master["Brightness Pristine"]
            master["Failure Time (s)"] = master["Time to Failure (ms)"] / 1000

            features = ["Board ID", "Pattern", "Solution", "Sensor", "Failure Time (s)", "Current", "Voltage",
                        "Brightness Difference"]

            # create a subset of the master data frame only with necessary features
            master_subset = master[features].copy()
            master_subset.dropna(inplace=True)

            # create label encoders for categorical variables
            le_pattern = LabelEncoder()
            le_solution = LabelEncoder()
            le_sensor = LabelEncoder()

            # apply label encoders, save true labels for plotting
            master_subset["True_Pattern"] = master_subset["Pattern"]
            master_subset["Pattern"] = le_pattern.fit_transform(master_subset["Pattern"])

            master_subset["True_Solution"] = master_subset["Solution"]
            master_subset["Solution"] = le_solution.fit_transform(master_subset["Solution"])

            master_subset["True_Sensor"] = master_subset["Sensor"]
            master_subset["Sensor"] = le_sensor.fit_transform(master_subset["Sensor"])

            # split data into X and Y
            X = master_subset[["Pattern", "Solution", "Sensor", "Failure Time (s)", "Voltage"]].copy()
            Y = master_subset[["Brightness Difference"]]

            # scale X using standard scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # record total intra-cluster variation
            ticv = []

            # test various values of K to find the best one
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, n_init=20)

                # fit kmeans algorithm
                kmeans.fit(X_scaled)

                # record total intra-cluster variation for K=k
                ticv.append(kmeans.inertia_)

            st.text("Elbow plot - used for finding the optimal number of clusters")

            # Plot elbow graph
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, 11), ticv, linewidth=4, marker="o", c="#5e2943")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Total Intra-Cluster Variation")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # choose 2 clusters based on elbow chart
            kmeans = KMeans(n_clusters=2, n_init=20)
            y_kmeans = kmeans.fit_predict(X_scaled)

            # add cluster column [0,1]
            master_subset["cluster"] = y_kmeans

            # ensure that the cluster labels stay the same each run
            mean_failure = master_subset.groupby("cluster")["Failure Time (s)"].mean()
            if mean_failure[0] > mean_failure[1]:
                # switch cluster labels
                master_subset["cluster"] = master_subset["cluster"].map({0: 1, 1: 0})

            # show data frame of cluster board ID's, patterns and sensors
            st.subheader("Cluster 0 Boards")
            st.dataframe(master_subset[master_subset["cluster"] == 0][["Board ID", "Pattern", "Sensor"]])

            st.subheader("Cluster 1 Boards")
            st.dataframe(master_subset[master_subset["cluster"] == 1][["Board ID", "Pattern", "Sensor"]])

            # apply PCA for visualization purposes
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            st.text("Data clustered using K-Means Clustering - visualized through PCA projection")

            # plot clusters using PCA projection
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap="managua")
            ax.set_xlabel("Principal Component #1")
            ax.set_ylabel("Principal Component #2")
            ax.set_title("K-Means Clustering (PCA Projection)")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # calculate cluster characteristics and counts for comparison
            cluster_statistics = master_subset.groupby("cluster")[["Failure Time (s)"]].mean()
            pattern_counts = master_subset.groupby(["cluster", "True_Pattern"]).size().unstack(fill_value=0)
            sensor_counts = master_subset.groupby(["cluster", "True_Sensor"]).size().unstack(fill_value=0)
            solution_counts = master_subset.groupby(["cluster", "True_Solution"]).size().unstack(fill_value=0)
            voltage_counts = master_subset.groupby(["cluster", "Voltage"]).size().unstack(fill_value=0)
            cluster_summary = master_subset.groupby("cluster")["Brightness Difference"].agg(["sum"])

            # color palette
            colors = ["#97e3fa", "#ffd584"]

            st.text("Cluster Summaries")

            # create figure and axes for bar plots of various cluster characteristics
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))

            # failure time by cluster
            ax = axes[0, 0]
            cluster_statistics["Failure Time (s)"].plot(kind="bar", ax=ax, color=colors)
            ax.set_title("Failure Time by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Time (seconds)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.grid(axis="y", alpha=0.3)

            # dendrite growth by cluster
            ax = axes[0, 1]
            cluster_summary["sum"].T.plot(kind="bar", ax=ax, color=colors, width=0.8)
            ax.set_xlabel("Cluster")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
            ax.set_ylim(0, 700)
            ax.set_ylabel("Dendrite Growth")
            ax.set_title("Dendrite Growth by Cluster")
            ax.grid(axis="y", alpha=0.3)

            # voltage by cluster
            ax = axes[1, 0]
            voltage_counts.T.plot(kind="bar", ax=ax, color=colors, width=0.8)
            ax.set_title("Voltage by Cluster")
            ax.set_xlabel("Voltage")
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(["Cluster 0", "Cluster 1"], loc="upper right")
            ax.grid(axis="y", alpha=0.3)

            # patterns by cluster
            ax = axes[1, 1]
            pattern_counts.T.plot(kind="bar", ax=ax, color=colors, width=0.8)
            ax.set_title("Pattern by Cluster")
            ax.set_xlabel("Pattern")
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(["Cluster 0", "Cluster 1"], loc="best")
            ax.grid(axis="y", alpha=0.3)

            # sensors by cluster
            ax = axes[2, 0]
            sensor_counts.T.plot(kind="bar", ax=ax, color=colors, width=0.8)
            ax.set_title("Sensor Type Distribution by Cluster")
            ax.set_xlabel("Sensor")
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(["Cluster 0", "Cluster 1"], loc="best")
            ax.grid(axis="y", alpha=0.3)

            # solutions by cluster
            ax = axes[2, 1]
            solution_counts.T.plot(kind="bar", ax=ax, color=colors, width=0.8)
            ax.set_title("Solution Type by Cluster")
            ax.set_xlabel("Solution")
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(["Cluster 0", "Cluster 1"], loc="best")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)


        cluster_analysis()

    # ------------------------------------------------------------------------------------------------------------------

    # pH plot ----------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("pH Plot")
        st.text("Plots failure time as a function of pH")


        def plot_ph():
            # get master data
            df = adds.get_master()

            # Remove solutions with no recorded Ph so they don't take up space in the legend
            df = df[(df["Solution"] == "Adipic Acid - 1.24mM") | (df["Solution"] == "Adipic Acid - 0.712mM") | (
                    df["Solution"] == "Adipic Acid - 0.388mM") | (df["Solution"] == "Succinic 0.388mM")]

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x="pH", y="Time to Failure (ms)", data=df, hue="Solution", ax=ax)

            ax.set_title("Time to Failure (ms) vs. pH by Solution Type")

            # Move legend to the right of the plot
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

            # Tight layout so the legend doesn't get cut off
            plt.tight_layout()
            st.pyplot(fig)


        plot_ph()

    # ------------------------------------------------------------------------------------------------------------------

    # CF/CV Plots ------------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("CF and CV Plots")
        st.text("Plots CF and CV Data")


        def CF_CV():

            # get CF and CV data
            CF = adds.get_master_cf_or_cv(cf_or_cv="CF")
            CV = adds.get_master_cf_or_cv(cf_or_cv="CV")

            # filter out files with bad data
            CF = CF[(CF["Capacitance (F)"] > 0) & (CF["Capacitance (F)"] < 100)]
            CV = CV[(CV["Capacitance (F)"] > 0) & (CV["Capacitance (F)"] < 100)]

            # list of sensor names and colors for plotting
            sensors = ["U1", "U2", "U3", "U4"]
            colors = ["c", "m", "y", "#47E183"]

            # take the average across all data for each sensor and frequency for CF data
            if CF is not None and not CF.empty:
                CF_average = CF.groupby(["Sensor", "Frequency (Hz)"]).agg({"Capacitance (F)": "mean", "Impedance (O)":
                    "mean", "Phase Angle (D)": "mean"}).reset_index()

            # take the average across all data for each sensor and voltage for CV data
            if CV is not None and not CV.empty:
                CV_average = CV.groupby(["Sensor", "Voltage (V)"]).agg(
                    {"Capacitance (F)": "mean", "Impedance (O)": "mean",
                     "Phase Angle (D)": "mean"}).reset_index()

            # create figure and subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))

            # flatten array
            ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

            # plot CF data - one line per sensor
            for i, sensor in enumerate(sensors):
                sensor_data = CF_average[CF_average["Sensor"] == sensor]

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
                sensor_data = CV_average[CV_average["Sensor"] == sensor]

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
    # ------------------------------------------------------------------------------------------------------------------

    # Current vs Time --------------------------------------------------------------------------------------------------
    with st.container(border=True):
        st.header("Current vs Time")
        st.text("Plots current as a function of time for each tested sensor separated by solution and board type")


        def current_vs_time():

            # Get joined data
            master_current_time = adds.get_master_current_time()

            # Add a unique sensor identifier
            master_current_time["Sensor ID"] = master_current_time["Board ID"] + "_" + master_current_time["Sensor"]

            # Plot data for each unique voltage
            for voltage in master_current_time["Voltage"].unique():

                st.text(f"Voltage: {voltage}")

                # Create a FacetGrid
                g = sns.FacetGrid(data=master_current_time[master_current_time["Voltage"] == voltage], row="Pattern",
                                  row_order=[1, 4, 7, 10], col="Solution",
                                  col_order=["DI Water", "Adipic Acid - 0.388mM",
                                             "Adipic Acid - 0.712mM", "Adipic Acid - 1.24mM", "Succinic 0.388mM",
                                             "Succinic 0.712 mM",
                                             "Succinic 1.425mM", "Succinic 3.6mM"], hue="Sensor",
                                  palette={"U1": "#FF0000", "U2": "#B6FF00",
                                           "U3": "#00FFFF", "U4": "#7F00FF"}, margin_titles=True, sharex=False,
                                  sharey=False)

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
    # ------------------------------------------------------------------------------------------------------------------

# option to view cached data frame
data = pd.read_csv("../master_cached.csv")
with st.expander("See Cached Data"):
    st.dataframe(data)
