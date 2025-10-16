# This is the first stage of the pipeline. Each function here will read and
# return a certain type of file in the data. This file standardizes reading each
# file, because there should be only one valid way to read data. Data loss
# should only occur here if it really needs to.

import pandas as pd
import numpy as np
import cv2
import typing
import os

# debug
# new method of reading data to avoid errors:
# get reads.py directory
reads_directory = os.path.dirname(os.path.abspath(__file__))

# get IDC_EM_Analysis directory
IDC_directory = os.path.dirname(reads_directory)

# Get the master data as a DataFrame with proper data types
# Future idea: Replace all occurrences of file names in cells with their
# DataFrame equivalent
def get_master():

    # read in masterlist
    csv_path = os.path.join(IDC_directory, "IDCSubmersionMasterlist_20250505.csv")
    master = pd.read_csv(csv_path)

    # Cast numeric columns to numbers
    numeric_cols = ["Voltage", "Pattern"]
    master[numeric_cols] = master[numeric_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    # Do not drop NaN rows, because these rows could be useful

    return master

# Get the last cached version of the result of adds.get_master()
def get_master_cached():
    try:
        # Read the file
        cached_path = os.path.join(IDC_directory, "master_cached.csv")
        return pd.read_csv(cached_path)

    except FileNotFoundError:
        # Return None if not readable
        return None

# Get a CurrentTime file as a DataFrame with proper data types
def get_current_time(file_name: str):
    try:
        # Read the file
        current_time = pd.read_csv(f"CurrentTime/{file_name}")
    except FileNotFoundError:
        # Return None if not readable
        return None

    # Cast all columns to numbers
    current_time = current_time.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    # Drop non-number rows
    current_time.dropna(inplace=True)

    return current_time


# Get a CF/CV file, (PRISTINE/EXPOSED), as a DataFrame with proper data types
def get_cf_or_cv(file_name: str):
    # Ensure file_name is string before using split
    if type(file_name) != str:
        return None

    # Get 7 string components of file name
    components = file_name.split(".")[0].split("_")
    # Return None if name is unconventional
    if len(components) != 7:
        return None

    # Get components needed for file path
    iteration = int(components[6])
    age = "PRISTINE" if iteration == 0 else "EXPOSED"
    cf_or_cv = components[5]

    try:
        # Read the file
        df = pd.read_csv(f"{cf_or_cv}/{cf_or_cv}_{age}/{file_name}")
    except FileNotFoundError:
        # Return None if not readable
        return None

    # Cast all columns to numbers
    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    # Drop non-number rows
    df.dropna(inplace=True)

    return df

# Get a board image from the file name
def get_board_image(file_name: str, age: typing.Literal["EXPOSED", "PRISTINE"]):
    # edit - original: file_path = f"Imgscans_{age}_edited/{file_name}"
    # file_path = f"../Imgscans_{age}/{file_name}"
    # Return None if file name is invalid
    # if file_name is np.nan or not os.path.isfile(file_path):
    # return None

    # edit: ensure correct folder is used based on PRISTINE/EXPOSED
    if age == "PRISTINE":
        directory = os.path.join(IDC_directory, "Imgscans_PRISTINE_templates")
    else:
        directory = os.path.join(IDC_directory, f"Imgscans_{age}")

    # Read and return file
    return cv2.imread(os.path.join(directory, file_name))

# Get a sensor image from the file name
def get_sensor_image(file_name: str, age: typing.Literal["EXPOSED", "PRISTINE"]):

    # edit: check if file name is invalid before continuing
    if file_name is np.nan or not isinstance(file_name, str):
        return None

    sensors_directory = os.path.join(IDC_directory, f"Imgscans_{age}_sensors")

    # Read and return file
    return cv2.imread(os.path.join(sensors_directory, file_name))
