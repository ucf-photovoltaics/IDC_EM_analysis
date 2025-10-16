import adds
import plotly.express as px

master = adds.get_master()

master.dropna(subset="Pattern", inplace=True)
master["Pattern"] = master["Pattern"].apply(int).apply(str)

# Melt the RGB columns
master = master.melt(
    id_vars=["Pattern", "Board ID", "Sensor"],
    value_vars=["R_PRISTINE", "G_PRISTINE", "B_PRISTINE", "R_EXPOSED", "G_EXPOSED", "B_EXPOSED"],
    var_name="Channel_Age",
    value_name="Value"
)

# Split "Channel_Age" into "Channel" and "Age"
master[["Channel", "Age"]] = master["Channel_Age"].str.extract(r"([RGB])_(PRISTINE|EXPOSED)")

# Pivot to get R, G, B in separate columns, with "Age" as one of the columns
master = master.pivot_table(
    index=["Pattern", "Board ID", "Sensor", "Age"],
    columns="Channel",
    values="Value"
).reset_index()

fig = px.scatter_3d(
    master,
    x="R",
    y="G",
    z="B",
    color="Pattern",
    symbol="Age",
    symbol_map={"PRISTINE": "circle-open", "EXPOSED": "circle"},
    opacity=0.6,
    hover_data=["Pattern", "Board ID", "Sensor"]
)

fig.show()
