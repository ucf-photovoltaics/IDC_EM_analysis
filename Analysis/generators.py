# A place for intensive code that cannot be reasonably ran each time the
# DataFrame is generated. This code should only be run occasionally when new
# data is added.

import reads
import cv2
import os
import typing

# Generate the cropped sensor images and store them
def gen_sensor_images():
    # Stores the coords as percentages of the sensor bounds
    # Sample use: pattern_to_sensor_to_coords[pattern][sensor]["x1"|"y2"...]

    #debug
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of current directory: {os.listdir('.')}")
    print(f"Contents of parent directory: {os.listdir('..')}")


    pattern_to_sensor_to_coords = {
        1: {
            "U1": {"x1":0.1012, "x2":0.3067, "y1":0.576, "y2":0.6601},
            "U2": {"x1":0.1377, "x2":0.2395, "y1":0.09905, "y2":0.2684},
            "U3": {"x1":0.7647, "x2":0.8325, "y1":0.07259, "y2":0.3275},
            "U4": {"x1":0.7736, "x2":0.8254, "y1":0.434, "y2":0.7743}
        },
        4: {
            "U1": {"x1":0.09917, "x2":0.3052, "y1":0.5761, "y2":0.6606},
            "U2": {"x1":0.1369, "x2":0.2389, "y1":0.09948, "y2":0.2688},
            "U3": {"x1":0.7653, "x2":0.8328, "y1":0.07529, "y2":0.3299},
            "U4": {"x1":0.773, "x2":0.8243, "y1":0.4364, "y2":0.7764}
        },
        7: {
            "U1": {"x1":0.1011, "x2":0.3066, "y1":0.5776, "y2":0.6608},
            "U2": {"x1":0.137, "x2":0.2387, "y1":0.1002, "y2":0.2691},
            "U3": {"x1":0.7646, "x2":0.8324, "y1":0.07282, "y2":0.3275},
            "U4": {"x1":0.7737, "x2":0.8255, "y1":0.4342, "y2":0.7739}
        },
        10: {
            "U1": {"x1":0.09913, "x2":0.3055, "y1":0.5753, "y2":0.6594},
            "U2": {"x1":0.1367, "x2":0.2384, "y1":0.0993, "y2":0.2683},
            "U3": {"x1":0.7644, "x2":0.8324, "y1":0.07448, "y2":0.3286},
            "U4": {"x1":0.7727, "x2":0.8243, "y1":0.4351, "y2":0.7745}
        }
    }

    # Helper function used to generate a singular sensor image and write to file,
    # given a master row and age
    def gen_sensor_image(master_row, age: typing.Literal["EXPOSED", "PRISTINE"]):

        # edit - original: matching_file_names = [file for file in os.listdir(f"Imgscans_{age}_edited")
        # if file.startswith(master_row["Board ID"])]
        #matching_file_names = [file for file in os.listdir(f"../Imgscans_{age}") if file.startswith(master_row["Board ID"])]

        # edit:
        # Get list of file names that start with this board ID
        print(f"Processing: {master_row['Board ID']}, {master_row['Sensor']}, {age}")

        # updated: use correct file path
        path = f"../Imgscans_PRISTINE_templates" if age == "PRISTINE" else f"../Imgscans_{age}"

        matching_file_names = [file for file in os.listdir(path) if file.startswith(master_row["Board ID"])]

        # If no files are found, return
        if len(matching_file_names) == 0:
            return
        
        # Take the first found file and use it
        # TODO Handle cases of boards having multiple scans, possibly by using the
        # iteration value
        file_name = matching_file_names[0]

        # Read board image
        board_img = reads.get_board_image(file_name, age)

        # Get width and height, to be used for calculating crop coords
        height, width, _ = board_img.shape
        # Calculate crop coords based on crop percentages
        coords = pattern_to_sensor_to_coords[master_row["Pattern"]][master_row["Sensor"]]
        x1 = round(coords["x1"] * width)
        x2 = round(coords["x2"] * width)
        y1 = round(coords["y1"] * height)
        y2 = round(coords["y2"] * height)

        # Get the cropped sensor
        sensor_image = board_img[y1:y2, x1:x2]
        
        # Get values to construct file name, and store file
        # TODO Add date. Date isn't used yet because of dates missing in master
        #month, day, year = tuple(map(int, master_row["Date"].split("/")))
        file_name = f"{master_row["Board ID"]}_{"000" if age == "PRISTINE" else "001"}_{master_row["Sensor"]}.jpg"

        # Write to file
        cv2.imwrite(f"../Imgscans_{age}_sensors/{file_name}", sensor_image)

    master = reads.get_master()

    # Drop NaNs because these columns are needed for creating the file name
    master.dropna(subset=["Board ID", "Date", "Sensor"], inplace=True)

    # Set index to a unique sensor identifier
    master.set_index(["Board ID", "Sensor"], inplace=True, drop=False)
    master.sort_index(inplace=True)

    # Drop duplicate indices
    master = master[~master.index.duplicated(keep='first')]

    # Counter variable for printed progress
    generated = 0

    for pristine_file_name in os.listdir("../Imgscans_PRISTINE_templates"):
        for sensor in ["U1", "U2", "U3", "U4"]:
            batch, pattern, id, _ = pristine_file_name.split("_")
            board_id = "_".join([batch, pattern, id])

            # Get master rows that match this file
            rows = master[(master["Board ID"] == board_id) & (master["Sensor"] == sensor)]

            # Skip if no master rows match
            if len(rows) == 0:
                continue

            # Generate image using the first found row
            gen_sensor_image(rows.iloc[0], "PRISTINE")
            
            # Show progress
            generated += 1
            print(f"Images generated: {generated}")

    # For each row in master, generate the EXPOSED image if possible
    for _, row in master.iterrows():
        gen_sensor_image(row, "EXPOSED")
        
        # Show progress
        generated += 1
        print(f"Images generated: {generated}")

# Run Area ---------------------------------------------------------------------
# Uncomment lines to run the generators

gen_sensor_images()
