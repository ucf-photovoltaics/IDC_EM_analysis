# This is the second stage of the pipeline. Additions are made to the data, such
# as new columns, and joins. Data removal should be minimal here, to prevent
# loss of useful data.

import reads
import pandas as pd
import numpy as np
import os
import typing
import cv2
import math

# Get the cleaned master data
def get_master(from_cache=True):
    # If cached version is requested, read and return cached version
    if from_cache:
        return reads.get_master_cached()

    # Read in data
    master = reads.get_master()
    
    # Add image file names
    for file_name in os.listdir("Imgscans_PRISTINE_sensors"):
        # Get components
        batch, pattern, id, _, sensor = file_name.split(".")[0].split("_")
        pattern = int(pattern)
        # Mask only where pattern and sensor match
        mask = (master["Pattern"] == pattern) & (master["Sensor"] == sensor)
        # Store the file name using the mask
        master.loc[mask, "Image_PRISTINE"] = file_name

    for file_name in os.listdir("Imgscans_EXPOSED_sensors"):
        # Get components
        batch, pattern, id, _, sensor = file_name.split(".")[0].split("_")
        board_id = "_".join([batch, pattern, id])
        # Mask only where board ID and sensor match
        mask = (master["Board ID"] == board_id) & (master["Sensor"] == sensor)
        # Store the file name using the mask
        master.loc[mask, "Image_EXPOSED"] = file_name
    # TODO Remove CV, CF, and CurrentTime file names from the stored CSV. Add
    # code here to populate those columns automatically

    # Get dendrite score of the exposed file, compared to the pristine
    def gen_dendrite_score(master_index, master_row):
        # Read images
        pristine_image = reads.get_sensor_image(master_row["Image_PRISTINE"], "PRISTINE")
        exposed_image = reads.get_sensor_image(master_row["Image_EXPOSED"], "EXPOSED")

        # Return NaN if images couldn't be read
        if pristine_image is None or exposed_image is None:
            # NaN is used instead of None because NaN is for numbers, None
            # is for objects
            return np.nan

        # Reorder from BGR to RGB
        pristine_image = cv2.cvtColor(pristine_image, cv2.COLOR_BGR2RGB)
        exposed_image = cv2.cvtColor(exposed_image, cv2.COLOR_BGR2RGB)

        # Split into RGB components
        r1, g1, b1 = cv2.split(pristine_image)
        r2, g2, b2 = cv2.split(exposed_image)

        # Convert rgb arrays into mean values
        r1, g1, b1 = np.mean(r1), np.mean(g1), np.mean(b1)
        r2, g2, b2 = np.mean(r2), np.mean(g2), np.mean(b2)

        # Generate and store score
        master.loc[master_index, "Dendrite Score"] = math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)

        # Store RGB values
        master.loc[master_index, "R_PRISTINE"] = r1
        master.loc[master_index, "G_PRISTINE"] = g1
        master.loc[master_index, "B_PRISTINE"] = b1
        master.loc[master_index, "R_EXPOSED"] = r2
        master.loc[master_index, "G_EXPOSED"] = g2
        master.loc[master_index, "B_EXPOSED"] = b2

        # Get brightness values for pristine and exposed images using the mean
        # of the RGB values and store in their own columns
        pristine_brightness=np.mean([r1, g1, b1])
        exposed_brightness=np.mean([r2, g2, b2])
        master.loc[master_index, "Brightness Pristine"]=pristine_brightness
        master.loc[master_index, "Brightness Exposed"]=exposed_brightness

    # Populate mean RGB and dendrite score
    for i, row in master.iterrows():
        gen_dendrite_score(i, row)

    return master

# Get a DataFrame that is the merging of the master data and all the
# CurrentTime files. Each row represents one current measurement at a given
# time, and it has data about the sensor, solution, etc.
def get_master_current_time():
    master = get_master()

    current_time_all = [] # List of all currentTime data frames

    # For each row, read the currentTime file if possible, and append it to list
    for _, row in master.iterrows():
        # Get the current_in value, which could be a float or file name
        current_in = row["Current"]

        # Read file, and skip if result is None
        current_time = reads.get_current_time(current_in)
        if current_time is None:
            continue

        # Add file_name column for joining purposes
        current_time["File Name"] = current_in

        # Append this current_time to current_time_all
        current_time_all.append(current_time)

    # Convert current_time_all, (a list), to a concatenation of all its contents
    current_time_all = pd.concat(current_time_all)

    # Join master with current_time_all
    master_current_time = master.merge(
        current_time_all,
        left_on="Current", right_on="File Name",
        how="inner"
    )

    # The file names are no longer needed
    master_current_time.drop(columns=["File Name", "Current"], inplace=True)

    return master_current_time

# Returns the master merged with all CF files, or all CV files. An "Age" column
# is added to differentiate "PRISTINE" vs "EXPOSED"
def get_master_cf_or_cv(cf_or_cv: typing.Literal["CF", "CV"]):
    master = get_master()

    df_all = [] # List of all DataFrames, either cf or cv, has an Age column

    # Populate df_all with both ages
    for age in ["PRISTINE", "EXPOSED"]:
        # For each row, read file, and append to list
        for _, row in master.iterrows():
            # Get file name
            baseline_or_post = "Baseline" if age == "PRISTINE" else "Post"
            file_name = row[f"{cf_or_cv}_{baseline_or_post}"]
            
            # Read df, skipping if result is None
            df = reads.get_cf_or_cv(file_name)
            if df is None:
                continue
            
            # Add Age column to differentiate PRISTINE and EXPOSED
            df["Age"] = age

            # Add File Name column for joining purposes
            df["File Name"] = file_name

            # Append this DataFrame to list
            df_all.append(df)

    # Convert lists to a concatenation of all their contents
    df_all = pd.concat(df_all, ignore_index=True)
    
    # Join master with baseline files
    master_pristine = master.merge(
        df_all,
        left_on=f"{cf_or_cv}_Baseline",
        right_on="File Name",
        how="inner"
    )
    # Join master with post files
    master_exposed = master.merge(
        df_all,
        left_on=f"{cf_or_cv}_Post",
        right_on="File Name",
        how="inner"
    )

    # Concat both merges into the result
    master_cf_or_cv = pd.concat([master_pristine, master_exposed])

    # The file names are no longer needed
    master_cf_or_cv.drop(columns=["File Name", f"{cf_or_cv}_Baseline", f"{cf_or_cv}_Post"], inplace=True)

    return master_cf_or_cv