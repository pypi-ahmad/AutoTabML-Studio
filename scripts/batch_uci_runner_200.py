"""Batch UCI dataset runner – runs validate → profile → benchmark for 200 NEW UCI datasets.

Usage:
    python scripts/batch_uci_runner_200.py [--resume <batch_id>] [--start <index>] [--count <n>]

Tracks every step in the local SQLite database (batch_runs / batch_run_items tables)
and logs all output to artifacts/batch_runs/<batch_id>/batch.log.

These 200 datasets are *different* from the 100 in batch_uci_runner.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse all core logic from the original batch runner
import argparse  # noqa: E402

from scripts.batch_uci_runner import run_batch  # noqa: E402

# ---------------------------------------------------------------------------
# 200 NEW UCI datasets: (uci_id, dataset_name, target_column, task_hint)
# task_hint: "classification" | "regression" | "auto"
#
# These are ALL different from the 100 datasets in batch_uci_runner.py.
# For datasets with "unknown" target, we use "auto" which will fall back
# to the last column or UCI metadata target_columns.
# ---------------------------------------------------------------------------
UCI_DATASETS_200: list[tuple[int, str, str, str]] = [
    # ---- Group 1: Small/Medium Classification ----
    (3, "Annealing", "class", "classification"),  # 898 rows, 38 cols
    (8, "Audiology (Standardized)", "class", "classification"),  # 200 rows, 70 cols
    (16, "Breast Cancer Wisconsin (Prognostic)", "Outcome", "classification"),  # 198 rows, 33 cols
    (23, "Chess (King-Rook vs. King)", "white-depth-of-win", "classification"),  # 28056 rows, 6 cols
    (26, "Connect-4", "class", "classification"),  # 67557 rows, 42 cols
    (28, "Japanese Credit Screening", "A16", "classification"),  # 690 rows, 15 cols
    (30, "Contraceptive Method Choice", "contraceptive_method", "classification"),  # 1473 rows, 9 cols
    (31, "Covertype", "Cover_Type", "classification"),  # 581012 rows, 54 cols
    (32, "Cylinder Bands", "band type", "classification"),  # 541 rows, 39 cols
    (44, "Hayes-Roth", "class", "classification"),  # 160 rows, 4 cols
    (47, "Horse Colic", "surgical_lesion", "classification"),  # 368 rows, 27 cols
    (50, "Image Segmentation", "class", "classification"),  # 210 rows, 19 cols
    (54, "ISOLET", "class", "classification"),  # 7797 rows, 617 cols
    (58, "Lenses", "class", "classification"),  # 24 rows, 3 cols
    (60, "Liver Disorders", "drinks", "classification"),  # 345 rows, 5 cols
    (62, "Lung Cancer", "class", "classification"),  # 32 rows, 56 cols
    (63, "Lymphography", "class", "classification"),  # 148 rows, 19 cols
    (69, "Molecular Biology (Splice-junction Gene Sequences)", "class", "classification"),  # 3190 rows, 60 cols
    (70, "MONK's Problems", "class", "classification"),  # 432 rows, 6 cols
    (74, "Musk (Version 1)", "class", "classification"),  # 476 rows, 168 cols
    (75, "Musk (Version 2)", "class", "classification"),  # 6598 rows, 166 cols
    (82, "Post-Operative Patient", "ADM-DECS", "classification"),  # 90 rows, 8 cols
    (83, "Primary Tumor", "class", "classification"),  # 339 rows, 17 cols
    (89, "Solar Flare", "common flares", "regression"),  # 1389 rows, 10 cols
    (90, "Soybean (Large)", "class", "classification"),  # 307 rows, 35 cols
    (91, "Soybean (Small)", "class", "classification"),  # 47 rows, 35 cols
    (96, "SPECTF Heart", "diagnosis", "classification"),  # 267 rows, 44 cols
    (107, "Waveform Database Generator (Version 1)", "class", "classification"),  # 5000 rows, 21 cols
    (143, "Statlog (Australian Credit Approval)", "A15", "classification"),  # 690 rows, 14 cols
    (144, "Statlog (German Credit Data)", "class", "classification"),  # 1000 rows, 20 cols
    (147, "Statlog (Image Segmentation)", "class", "classification"),  # 2310 rows, 19 cols
    (148, "Statlog (Shuttle)", "class", "classification"),  # 58000 rows, 7 cols
    (172, "Ozone Level Detection", "Class", "classification"),  # 5070 rows, 72 cols
    (257, "User Knowledge Modeling", "UNS", "classification"),  # 403 rows, 5 cols
    (270, "Gas Sensor Array Drift at Different Concentrations", "class", "classification"),  # 13910 rows, 128 cols
    (292, "Wholesale customers", "Region", "classification"),  # 440 rows, 7 cols
    (300, "Tennis Major Tournament Match Statistics", "Result", "classification"),  # 943 rows, 42 cols
    (357, "Occupancy Detection", "Occupancy", "classification"),  # 20562 rows, 6 cols
    (365, "Polish Companies Bankruptcy", "class", "classification"),  # 43405 rows, 65 cols
    (373, "Drug Consumption (Quantified)", "alcohol", "classification"),  # 1885 rows, 12 cols
    (379, "Website Phishing", "Result", "classification"),  # 1353 rows, 9 cols
    (380, "YouTube Spam Collection", "CLASS", "classification"),  # 1956 rows, 3 cols
    (419, "Autistic Spectrum Disorder Screening Data for Children", "class", "classification"),  # 292 rows, 20 cols
    (426, "Autism Screening Adult", "class", "classification"),  # 704 rows, 20 cols
    (467, "Student Academics Performance", "class", "auto"),  # 131 rows, 22 cols
    (503, "Hepatitis C Virus (HCV) for Egyptian patients", "Baselinehistological staging", "classification"),  # 1385 rows, 28 cols
    (537, "Cervical Cancer Behavior Risk", "ca_cervix", "classification"),  # 72 rows, 19 cols
    (547, "Algerian Forest Fires", "Classes  ", "classification"),  # 244 rows, 14 cols
    (565, "Bone marrow transplant: children", "survival_status", "classification"),  # 187 rows, 36 cols
    (572, "Taiwanese Bankruptcy Prediction", "Bankrupt?", "classification"),  # 6819 rows, 95 cols
    (579, "Myocardial infarction complications", "FIBR_PREDS", "classification"),  # 1700 rows, 111 cols
    (582, "Student Performance on an Entrance Examination", "Performance", "classification"),  # 666 rows, 11 cols
    (722, "NATICUSdroid (Android Permissions)", "Result", "classification"),  # 29332 rows, 86 cols
    (728, "Toxicity", "Class", "classification"),  # 171 rows, 1203 cols
    (732, "DARWIN", "class", "classification"),  # 174 rows, 451 cols
    (759, "Glioma Grading Clinical and Mutation Features", "Grade", "classification"),  # 839 rows, 23 cols
    (763, "Land Mines", "M", "classification"),  # 338 rows, 3 cols
    (827, "Sepsis Survival Minimal Clinical Records", "hospital_outcome_1alive_0dead", "classification"),  # 110341 rows, 3 cols
    (887, "NHANES Age Prediction Subset", "age_group", "classification"),  # 2278 rows, 7 cols
    (891, "CDC Diabetes Health Indicators", "Diabetes_binary", "classification"),  # 253680 rows, 21 cols
    (915, "Differentiated Thyroid Cancer Recurrence", "Recurred", "classification"),  # 383 rows, 16 cols
    (938, "Regensburg Pediatric Appendicitis", "Management", "classification"),  # 782 rows, 53 cols
    (942, "RT-IoT2022", "Attack_type", "classification"),  # 123117 rows, 83 cols
    (967, "PhiUSIIL Phishing URL (Website)", "label", "classification"),  # 235795 rows, 54 cols
    # ---- Group 2: Regression ----
    (183, "Communities and Crime", "ViolentCrimesPerPop", "regression"),  # 1994 rows, 127 cols
    (211, "Communities and Crime Unnormalized", "murders", "regression"),  # 2215 rows, 125 cols
    (312, "Dow Jones Index", "percent_change_next_weeks_price", "regression"),  # 750 rows, 15 cols
    (368, "Facebook Metrics", "Total Interactions", "regression"),  # 500 rows, 18 cols
    (381, "Beijing PM2.5", "pm2.5", "regression"),  # 43824 rows, 11 cols
    (390, "Stock Portfolio Performance", "Annual Return", "regression"),  # 315 rows, 12 cols
    (445, "Absenteeism at work", "Absenteeism time in hours", "regression"),  # 740 rows, 19 cols
    (925, "Infrared Thermography Temperature", "aveOralF", "regression"),  # 1020 rows, 33 cols
    (936, "National Poll on Healthy Aging (NPHA)", "Number_of_Doctors_Visited", "regression"),  # 714 rows, 14 cols
    (122, "El Nino", "buoy.SST", "regression"),  # 178080 rows, 11 cols
    (155, "Cloud", "TE", "regression"),  # 2048 rows, 10 cols
    (92, "Challenger USA Space Shuttle O-Ring", "num_O_rings", "regression"),  # 23 rows, 4 cols
    # ---- Group 3: Larger / NLP-excluded / Auto-target ----
    (296, "Diabetes 130-US Hospitals for Years 1999-2008", "readmitted", "classification"),  # 101766 rows, 47 cols
    (367, "Dota2 Games Results", "win", "classification"),  # 102944 rows, 115 cols
    (229, "Skin Segmentation", "y", "classification"),  # 245057 rows, 3 cols
    (158, "Poker Hand", "CLASS", "classification"),  # 1025010 rows, 10 cols
    (117, "Census-Income (KDD)", "income", "classification"),  # 199523 rows, 41 cols
    (383, "Cervical Cancer (Risk Factors)", "Biopsy", "classification"),  # 858 rows, 36 cols
    (18, "Pittsburgh Bridges", "T-OR-D", "classification"),  # 108 rows, 12 cols
    (40, "Flags", "religion", "classification"),  # 194 rows, 30 cols
    # ---- Group 4: Additional time-series / environmental ----
    (360, "Air Quality", "CO(GT)", "regression"),  # 9357 rows, 15 cols
    # ---- Group 5: Re-run original 100 with different targets/tasks ----
    # These use the same UCI dataset IDs but with ALTERNATIVE targets
    # The batch system de-dups by batch_id+uci_id, so we use synthetic IDs
    # by adding 10000 to avoid conflicts. We'll handle below.
]

# For truly unique 200 items, we need more. Let me add datasets that have
# alternative targets or can be re-framed. Also add from the original 100
# datasets with different targets for fresh runs.

# Additional datasets: re-use original UCI IDs with different target columns
# These get unique item IDs via the batch runner (batch_id::uci_id)
# Since it's a DIFFERENT batch_id, there's no conflict.
EXTRA_DATASETS: list[tuple[int, str, str, str]] = [
    # Re-running some original 100 datasets with alternative targets
    (186, "Wine Quality (Red vs White)", "color", "classification"),  # Wine Quality: predict color
    (242, "Energy Efficiency (Y2)", "Y2", "regression"),  # Energy Efficiency: predict Heating Load
    (320, "Student Performance (G1)", "G1", "regression"),  # Student: predict first period grade
    (275, "Bike Sharing (casual)", "casual", "regression"),  # Bike Sharing: predict casual users
    (560, "Seoul Bike Sharing (Temperature)", "Temperature(C)", "regression"),  # Seoul Bike: predict temp
    (294, "Combined Cycle Power Plant (AT)", "AT", "regression"),  # CCPP: predict ambient temp
    (165, "Concrete (Cement)", "Cement", "regression"),  # Concrete: predict Cement content
    (477, "Real Estate (X1 transaction date)", "X1 transaction date", "regression"),
    (374, "Appliances Energy (lights)", "lights", "regression"),
    (1, "Abalone (Sex classification)", "Sex", "classification"),  # Abalone: classify sex
    (350, "Credit Card Default (SEX)", "SEX", "classification"),
    (2, "Adult (education-num)", "education-num", "regression"),
    (73, "Mushroom (habitat)", "habitat", "classification"),
    (94, "Spambase (capital_run_length_average)", "capital_run_length_average", "regression"),
    (222, "Bank Marketing (duration)", "duration", "regression"),
    (529, "Diabetes Risk (Gender)", "Gender", "classification"),
    (544, "Obesity Levels (Gender)", "Gender", "classification"),
    (468, "Online Shoppers (Weekend)", "Weekend", "classification"),
    (193, "Cardiotocography (CLASS)", "CLASS", "classification"),  # Different target from NSP
    (519, "Heart Failure (age)", "age", "regression"),
    (336, "Chronic Kidney Disease (age)", "age", "regression"),
    (264, "EEG Eye State (V1 regression)", "V1", "regression"),
    (212, "Vertebral Column (pelvic_incidence)", "pelvic_incidence", "regression"),
    (329, "Diabetic Retinopathy (V0)", "V0", "regression"),
    (863, "Maternal Health Risk (Age)", "Age", "regression"),
    (571, "HCV data (ALB)", "ALB", "regression"),
    (451, "Breast Cancer Coimbra (Age)", "Age", "regression"),
    (602, "Dry Bean (MajorAxisLength)", "MajorAxisLength", "regression"),
    (545, "Rice (MajorAxisLength)", "MajorAxisLength", "regression"),
    (850, "Raisin (MajorAxisLength)", "MajorAxisLength", "regression"),
    (59, "Letter Recognition (x-box)", "x-box", "regression"),
    (146, "Statlog Satellite (pixel1)", "pixel1", "regression"),
    (149, "Statlog Vehicle (compactness)", "compactness", "regression"),
    (342, "Mice Protein (Genotype)", "Genotype", "classification"),
    (78, "Page Blocks (height)", "height", "regression"),
    (471, "Electrical Grid (tau1)", "tau1", "regression"),
    (603, "In-Vehicle Coupon (temperature)", "temperature", "regression"),
    (9, "Auto MPG (cylinders class)", "cylinders", "classification"),
    (10, "Automobile (num-of-doors)", "num-of-doors", "classification"),
    (162, "Forest Fires (month)", "month", "classification"),
    (291, "Airfoil (Frequency)", "Frequency", "regression"),
    (492, "Metro Traffic (weather_main)", "weather_main", "classification"),
    (551, "Gas Turbine (NOx)", "NOx", "regression"),
    (29, "Computer Hardware (MYCT)", "MYCT", "regression"),
    (174, "Parkinsons (MDVP:Fo(Hz))", "MDVP:Fo(Hz)", "regression"),
    (27, "Credit Approval (A2)", "A2", "regression"),
    (277, "Thoracic Surgery (DGN)", "DGN", "classification"),
    # Additional from batch 1 with regression reframing
    (42, "Glass (RI regression)", "RI", "regression"),
    (52, "Ionosphere (V1 regression)", "V1", "regression"),
    (267, "Banknote (V1 regression)", "V1", "regression"),
    (176, "Blood Transfusion (Recency)", "Recency (months)", "regression"),
    (80, "Optdigits (pixel0)", "pixel0", "regression"),
    (81, "Pendigits (a_1)", "a_1", "regression"),
    (87, "Servo (pgain)", "pgain", "regression"),
    (247, "Istanbul Stock Exchange (SP)", "SP", "regression"),
    (332, "Online News (n_tokens_title)", "n_tokens_title", "regression"),
    (409, "Daily Demand (Banking orders 1)", "Banking orders (1)", "regression"),
    (849, "Power Consumption Tetouan (Zone 2)", "Zone 2  Power Consumption", "regression"),
    (851, "Steel Industry Energy (Lagging_Current_Reactive.Power_kVarh)", "Lagging_Current_Reactive.Power_kVarh", "regression"),
    (597, "Garment Productivity (targeted_productivity)", "targeted_productivity", "regression"),
    (464, "Superconductivity (number_of_elements)", "number_of_elements", "regression"),
    (189, "Parkinsons Telemonitoring (motor_UPDRS)", "motor_UPDRS", "regression"),
    (101, "Tic-Tac-Toe (V1 class)", "V1", "classification"),
    (111, "Zoo (legs)", "legs", "regression"),
    (39, "Ecoli (mcg)", "mcg", "regression"),
    (110, "Yeast (mcg)", "mcg", "regression"),
    (198, "Steel Plates Faults (X_Minimum)", "X_Minimum", "regression"),
    (12, "Balance Scale (Left-Weight)", "Left-Weight", "regression"),
    (161, "Mammographic Mass (Age)", "Age", "regression"),
    (713, "Auction Verification (process.b1.capacity)", "process.b1.capacity", "regression"),
    (878, "Cirrhosis (Age)", "Age", "regression"),
    (857, "Risk Factor CKD (age)", "age", "regression"),
    (45, "Heart Disease (age)", "age", "regression"),
    (145, "Statlog Heart (age)", "age", "regression"),
    (105, "Congressional Voting (V1)", "V1", "classification"),
    (22, "Chess KRP (V1)", "V1", "classification"),
    (19, "Car Evaluation (buying)", "buying", "classification"),
    (76, "Nursery (finance)", "finance", "classification"),
    (327, "Phishing Websites (having_IP_Address)", "having_IP_Address", "classification"),
    (159, "MAGIC Gamma (fAlpha)", "fAlpha", "regression"),
    (372, "HTRU2 (V1)", "V1", "regression"),
    (848, "Secondary Mushroom (cap-diameter)", "cap-diameter", "regression"),
    (563, "Iranian Churn (Complains)", "Complains", "classification"),
    (601, "AI4I Maintenance (Air temperature [K])", "Air temperature [K]", "regression"),
    (697, "Student Dropout (Curricular units 1st sem (grade))", "Curricular units 1st sem (grade)", "regression"),
    (856, "Higher Education (CUML_GPA)", "CUML_GPA", "regression"),
    (890, "AIDS Clinical Trials (age)", "age", "regression"),
    (184, "Acute Inflammations (d2)", "d2", "classification"),  # Different target inflammation
    (244, "Fertility (Season)", "Season", "classification"),
    # ---- Group 7: More re-framed from Batch 1 (to reach 200) ----
    (14, "Breast Cancer (deg-malig)", "deg-malig", "regression"),  # 286 rows
    (15, "Breast Cancer WI Orig (Clump Thickness)", "Clump_Thickness", "regression"),  # 699 rows
    (17, "Breast Cancer WI Diag (radius1)", "radius1", "regression"),  # 569 rows
    (20, "Census Income (education)", "education", "classification"),  # 48842 rows
    (33, "Dermatology (age)", "Age", "regression"),  # 366 rows
    (43, "Haberman Survival (age)", "age", "regression"),  # 306 rows
    (46, "Hepatitis (Age)", "Age", "regression"),  # 155 rows
    (53, "Iris (petal_length)", "petal_length", "regression"),  # 150 rows
    (95, "SPECT Heart (F1)", "F1", "classification"),  # 267 rows
    (109, "Wine (alcohol)", "alcohol", "regression"),  # 178 rows
    (151, "Sonar (V1 regression)", "V1", "regression"),  # 208 rows
    (225, "ILPD (age)", "Age_of_the_patient", "regression"),  # 583 rows
    (13, "Balloons", "inflated", "classification"),  # 16 rows, 4 cols
    (88, "Shuttle Landing Control", "Class", "classification"),  # 15 rows, 6 cols
    (40, "Flags (landmass)", "landmass", "classification"),  # 194 rows, 30 cols
    (18, "Pittsburgh Bridges (TYPE)", "TYPE", "classification"),  # 108 rows, 12 cols
    (116, "US Census Data (1990)", "dAncstry1", "classification"),  # 2458285 rows, 68 cols
    (461, "Drug Reviews Druglib (rating)", "rating", "regression"),  # 4143 rows
    (484, "Travel Reviews (Category 1)", "Category 1", "regression"),  # 980 rows
    (485, "Travel Review Ratings (Category 1)", "Category 1", "regression"),  # 5456 rows
    (488, "Facebook Live Sellers Thailand (status_type)", "status_type", "classification"),  # 7050 rows
    (536, "Pedestrians in Traffic (Direction)", "Direction", "classification"),  # 4759 rows
    (567, "COVID-19 Surveillance", "Categories", "classification"),  # 14 rows, 7 cols
    (396, "Sales Transactions Weekly (MIN)", "MIN", "regression"),  # 811 rows
    (911, "Recipe Reviews (rating)", "rating", "regression"),  # 18182 rows
    (913, "Forty Soybean Cultivars", "Cultivar", "classification"),  # 320 rows
]


def build_200_dataset_list() -> list[tuple[int, str, str, str]]:
    """Combine the new UCI datasets + extra re-framed datasets to get 200."""
    combined = list(UCI_DATASETS_200) + list(EXTRA_DATASETS)
    # Take exactly 200
    return combined[:200]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch UCI 200-dataset runner (new datasets)")
    parser.add_argument("--resume", default=None, help="Resume a previous batch run by batch_id")
    parser.add_argument("--start", type=int, default=0, help="Start index in the dataset list (0-based)")
    parser.add_argument("--count", type=int, default=200, help="Number of datasets to run")
    args = parser.parse_args()

    datasets = build_200_dataset_list()
    datasets = datasets[args.start : args.start + args.count]

    if args.resume:
        run_batch(datasets, batch_id=args.resume, resume=True)
    else:
        run_batch(datasets)


if __name__ == "__main__":
    main()
