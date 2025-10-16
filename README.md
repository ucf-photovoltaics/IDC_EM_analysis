# P07 DARTS Analysis of Multimodal IDC Sensor Data

## Description
The purpose of this repository is for training purposes apart of UCF Data-Enabled Research Training and Development Sprint of summer 2025. This repository contains csv data and image files collected from testing of interdigitated Comb sensors for failure analysis. Students will use this dataset with python-based analysis techniques to explore the relationships between performance, degradation, and experimental conditions. Students will use these images with python-based imaging techniques for validating electrochemical degradation patterns observed in sensor data.

## Structure
📦darts_idc_analysis<br>
 ┣ 📂Analysis _- shared tools and plots for data analysis_<br>
 ┣ 📂CF _- capacitance vs frequencey csv files_<br>
 ┃ ┣ 📂CF_EXPOSED _- csv files for IDC boards that have been biased and exposed_<br>
 ┃ ┗ 📂CF_PRISTINE _- csv files for fresh IDC boards not exposed to elements_<br>
 ┣ 📂CurrentTime _- some tests have variable current over time, and the csv files are here_<br>
 ┣ 📂CV _- capacitance vs voltage csv files_<br>
 ┃ ┣ 📂CV_EXPOSED _- csv files for IDC boards that have been biased and exposed_<br>
 ┃ ┗ 📂CV_PRISTINE _- csv files for fresh IDC boards not exposed to elements_<br>
 ┣ 📂Imgscans_EXPOSED _- IDC images that have been exposed_<br>
 ┣ 📂Imgscans_PRISTINE _- image scans of pristine boards_<br>
 ┣ 📜IDCSubmersionMasterlist_20250505.csv _- central reference list of IDC with information including experimental conditions, image filenames, and test parameters_<br>
 ┣ 📜.gitignore<br>
 ┗ 📜README.md
 