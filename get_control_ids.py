'''
This script finds the IDs for age and sex-matched controls from both discovery and
validation cohorts.

SET the following before running.
- AD_METADATA_FILE:  file containing metadata of the AD samples used for the analysis.
- CONTRIBUTOR: which contributors to use (applicable to GNPC data)
- TARGET_CONTROL_N: number of controls to extract.
- control_meta_file: the file to save the control IDs.
'''

import pandas as pd
import numpy as np

cohort = "val" # either "dis" (for discovery) or "val" (for validation)

if cohort == "dis":
    AD_METADATA_FILE = "discovery_metadata.csv"
    CONTRIBUTOR = "F"
    control_meta_file = "metadata_controls_selected_discovery_AGEMATCHED.csv"
    TARGET_CONTROL_N = 1000 
else:
    AD_METADATA_FILE = "validation_metadata.csv"
    CONTRIBUTOR = ["A", "C", "D", "G", "L", "Q", "R"]
    control_meta_file = "metadata_controls_selected_validation_all_AGEMATCHED.csv"
    TARGET_CONTROL_N = 1000

# --- Configuration ---
RAW_CLINICAL_FILE = "~/files/GNPC_HDS_general/HDS V1.3/ClinicalV1_3.csv"
RAW_PROT_META_FILE = "~/files/GNPC_HDS_general/HDS V1.3/ProteomicsMetaV1_3.csv"
CODEBREAK_FILE = "~/files/GNPC_HDS_general/HDS V1.3/PersonMappingV1_3.csv"

# --- 1. Load the AD Cohort (The Target Distribution) ---
print("Loading AD Cohort demographics...")
ad_df = pd.read_csv(AD_METADATA_FILE)

# Create Age Bins for stratification (e.g., <60, 60-70, 70-80, 80+)
# bins = [0, 60, 70, 80, 120]
# labels = ['<60', '60-70', '70-80', '80+']
# Using 5-year increments forces a much tighter mean match
bins = [0, 60, 65, 70, 75, 80, 85, 90, 120]
labels = ['<60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90+']

ad_df['age_bin'] = pd.cut(ad_df['age_at_visit'], bins=bins, labels=labels)
# In ad_df, we have 33, 116, 121, and 108 for the bins in labels order

# Calculate the Target Proportions
stratification_target = ad_df.groupby(['sex', 'age_bin'], observed=False).size() / len(ad_df)
print("\nTarget Demographics (AD Cohort):")
print(stratification_target)

# --- 2. Load and Prepare the Control Pool ---
print("\nLoading Raw Data to find Controls...")
clin_df = pd.read_csv(RAW_CLINICAL_FILE)
prot_meta_df = pd.read_csv(RAW_PROT_META_FILE)
codebreak_df = pd.read_csv(CODEBREAK_FILE)

# Merge
pool_df = pd.merge(clin_df, prot_meta_df, on="sample_id")
pool_df = pd.merge(pool_df, codebreak_df, on="sample_id")

if cohort == "dis":
    c_cond = (pool_df['contributor_code_y'] == CONTRIBUTOR)
else:
    c_cond = (pool_df['contributor_code_y'].isin(CONTRIBUTOR))

# Filter for HEALTHY CONTROLS from Contributor F, Visit 1
# Definition of Control: No AD, No PD, No FTD, No ALS
if cohort == "dis":
    control_pool = pool_df[
        (pool_df['row_check'] == 'PASS') &
        c_cond &
        (pool_df['visit'] == 1) &
        (pool_df['sample_matrix'] == 'EDTA Plasma') &
        (pool_df['ad'] == 0) &
        (pool_df['pd'] == 0) &
        (pool_df['ftd'] == 0) &
        (pool_df['als'] == 0)
    ].copy()
else:
    control_pool = pool_df[
    (pool_df['row_check'] == 'PASS') &
    c_cond &
    (pool_df['visit'] == 1) &
    (pool_df['sample_matrix'] == 'EDTA Plasma') &
    (pool_df['ad'] != 1) &
    (pool_df['pd'] != 1) &
    (pool_df['ftd'] != 1) &
    (pool_df['als'] != 1)
].copy()

# Apply the same Age Binning to the controls
control_pool['age_bin'] = pd.cut(control_pool['age_at_visit'], bins=bins, labels=labels)

print(f"Found {len(control_pool)} total available controls (Contributor {CONTRIBUTOR}).")

# --- 3. Perform Stratified Sampling ---
selected_control_ids = []

# stratified random sampling
print(f"\nSampling {TARGET_CONTROL_N} controls to match AD demographics...")
for (sex, age_bin), proportion in stratification_target.items():
    # How many controls do we need for this specific bin?
    n_needed = int(np.round(proportion * TARGET_CONTROL_N))
    
    if n_needed == 0:
        continue
        
    # Find available controls in this bin
    available_controls = control_pool[
        (control_pool['sex'] == sex) & 
        (control_pool['age_bin'] == age_bin)
    ]
    
    # Sample
    if len(available_controls) >= n_needed:
        # We have enough controls so we pick random subset
        selected = available_controls.sample(n=n_needed, random_state=42)
        status = "Matched"
    else:
        # We don't have enough -> Take everyone we have
        selected = available_controls
        status = f"Under-sampled (Wanted {n_needed}, Got {len(selected)})"
        
    print(f"  Sex: {sex}, Age: {age_bin} -> {status}")
    selected_control_ids.extend(selected['sample_id'].tolist())

breakpoint()

# --- 4. Save Result ---
final_controls_df = control_pool[control_pool['sample_id'].isin(selected_control_ids)]
final_controls_df.to_csv(control_meta_file, index=False)
print(f"\nSuccessfully selected {len(final_controls_df)} controls.")
print(f"\nSaved to {control_meta_file}")

# Quick check of the result
print("\nFinal Control Age Mean:", final_controls_df['age_at_visit'].mean())
print("Original AD Age Mean:  ", ad_df['age_at_visit'].mean())
