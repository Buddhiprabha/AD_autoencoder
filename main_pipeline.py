'''
This script runs the main pipeline of the analysis.

SET the following variables before running.

- best_model_folder: folder containing the best model folder from autoencoder training
- image_folder: to store all output images
- OUT_DIS_PATHWAYS: to store pathway and differential expression files of the discovery cohort
- OUT_VAL_PROJ_PATHWAYS: to store pathway and differential expression files of the validation cohort
- OUT_VAL_INDEP_PATHWAYS: to store pathway and differential expression files of the independent run of 
                          the validation cohort
- TRAIN_PARQUET: parquet file containing proteomic data for the discovery cohort AD samples
- METADATA_FILE: metadata file in CSV format containing patient metadata for discovery cohort
- DIS_CTRL_PROT: parquet file containing proteomic data for the discovery cohort controls
- DIS_CTRL_META: metadata file in CSV format containing control metadata for discovery cohort
- DIS_BATCH_COL: column containing batch effect data in the discovery cohort

- VAL_PARQUET: parquet file containing proteomic data for the validation cohort AD samples
- VAL_METADATA: metadata file in CSV format containing patient metadata for the validation cohort
- VAL_CTRL_PROT: parquet file containing proteomic data for the validation cohort controls
- VAL_CTRL_META: metadata file in CSV format containing control metadata for validation cohort
- VAL_BATCH_COL: column containing batch effect data in the validation cohort

'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import gseapy as gp
import os
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Import plotting functions
from plotting import plot_latent_alignment, generate_discovery_plots, generate_validation_plots

# the following variable can be set to True to flip the cohorts(i.e., multi-site as discovery and F as validaiton)
is_sensitivity = False 


if is_sensitivity:
    best_model_folder = "sensitivity_model_folder_name"
else:
    best_model_folder = "best_model_folder_name"

# Setup Output Folders
if is_sensitivity:
    image_folder = "clust_sens" 
    OUT_DIS_PATHWAYS = "sensitivity_pathways_discovery"
    OUT_VAL_PROJ_PATHWAYS = "sensitivity_pathways_val_projected"
    OUT_VAL_INDEP_PATHWAYS = "sensitivity_pathways_val_independent"
else:
    image_folder = "clust_all"
    OUT_DIS_PATHWAYS = "pathways_discovery"
    OUT_VAL_PROJ_PATHWAYS = "pathways_val_projected"
    OUT_VAL_INDEP_PATHWAYS = "pathways_val_independent"

os.makedirs(image_folder, exist_ok=True)

TRAIN_LATENT    = f'{best_model_folder}/latent_proteomics_subtypes.csv'
ADJACENCY_FILE  = f'{best_model_folder}/training_adjacency_matrix.npy'
ENCODER_WEIGHTS = f'{best_model_folder}/best_gae_encoder.weights.h5'

# ---- Input Files ----
if is_sensitivity:
    # DISCOVERY: Multi-Site
    TRAIN_PARQUET    = 'validation_AD.parquet'
    METADATA_FILE    = 'metadata_validation_AD.csv'
    DIS_CTRL_PROT    = 'validation_controls.parquet'
    DIS_CTRL_META    = 'metadata_validation_controls.csv'
    DIS_BATCH_COL    = 'val_batch_col'
    
    # VALIDATION: Cohort F
    VAL_PARQUET      = 'discovery_AD.parquet'
    VAL_METADATA     = 'metadata_discovery_AD.csv'
    VAL_CTRL_PROT    = 'discovery_controls.parquet'
    VAL_CTRL_META    = 'metadata_discovery_controls.csv'
    VAL_BATCH_COL    = 'dis_batch_col'
    
else:
    # DISCOVERY: Cohort F
    TRAIN_PARQUET    = 'discovery_AD.parquet'
    METADATA_FILE    = 'metadata_discovery_AD.csv'
    DIS_CTRL_PROT    = 'discovery_controls.parquet'
    DIS_CTRL_META    = 'metadata_discovery_controls.csv'
    DIS_BATCH_COL    = 'dis_batch_col'
    
    # VALIDATION: Multi-Site
    VAL_PARQUET      = 'validation_AD.parquet'
    VAL_METADATA     = 'metadata_validation_AD.csv'
    VAL_CTRL_PROT    = 'validation_controls.parquet'
    VAL_CTRL_META    = 'metadata_validation_controls.csv'
    VAL_BATCH_COL    = 'val_batch_col'

df_apoe      = pd.read_csv("~/files/GNPC_HDS_general/HDS V1.3/GeneticsV1_3.csv")
ANALYTE_FILE = '~/files/GNPC_HDS_general/HDS V1.3/AnalyteInfoV1_3.csv'

ID_COL         = 'sample_id'
CONTRIB_COL    = 'contributor_code_y'
COGNITIVE_COL  = 'cognitive_test_score'
AGE_COL        = 'age_at_visit'
SEX_COL        = 'sex'

N_CLUSTERS   = 3
RANDOM_STATE = 42

# =========================================================
# 1. AUTOENCODER MODEL CLASSES
# =========================================================
def build_encoder(num_nodes, latent_dim, funnel_dim, drop_rate):
    x_in  = Input(shape=(num_nodes, 1), name="Input_Features")
    hflat = layers.Flatten()(x_in)
    h     = layers.Dense(funnel_dim, activation='relu')(hflat)
    h     = layers.Dropout(drop_rate)(h)
    z     = layers.Dense(latent_dim, name="Latent_Z")(h)
    return Model(x_in, z, name="Encoder")

# =========================================================
# 2. LOAD DISCOVERY DATA
# =========================================================
print("\n=== STEP 2: Load training data ===")
df_train = pd.read_parquet(TRAIN_PARQUET)
df_train_meta = pd.read_csv(METADATA_FILE)

# Add APOE data safely
df_train_meta = pd.merge(df_train_meta, df_apoe, on=ID_COL, how='left')

# Binarize APOE Genotype ---
apoe_col = 'variant' 

if apoe_col in df_train_meta.columns:
    def determine_apoe4_status(val):
        if val in [34, 44, 24]: return 1
        elif val in [33, 23, 22]: return 0
        else: return np.nan 

    df_train_meta['apoe4_status'] = df_train_meta[apoe_col].apply(determine_apoe4_status)

# Filter by MMSE for early-stage AD (15 <= MMSE <= 30)
df_train_meta[COGNITIVE_COL] = pd.to_numeric(df_train_meta[COGNITIVE_COL], errors='coerce')
mask_train_valid = (df_train_meta[COGNITIVE_COL] >= 15) & (df_train_meta[COGNITIVE_COL] <= 30)

# <--- NEW: STRICT LATE-ONSET FILTER FOR DISCOVERY --->
df_train_meta[AGE_COL] = pd.to_numeric(df_train_meta[AGE_COL], errors='coerce')
mask_train_load = df_train_meta[AGE_COL] >= 65.0

# Calculate sample sizes to see the effect
original_train_n = mask_train_valid.sum()
mask_train_valid = mask_train_valid & mask_train_load
final_train_n = mask_train_valid.sum()

print(f"Discovery Cohort size before age filter: {original_train_n}")
print(f"Discovery Cohort size after age >= 65 filter: {final_train_n}")
print(f"-> Dropped {original_train_n - final_train_n} early-onset patients from Discovery (Cohort F).")

df_train_meta = df_train_meta[mask_train_valid].set_index(ID_COL)

# Load latent space and find intersection (already outlier-free)
df_train_latent = pd.read_csv(TRAIN_LATENT, index_col=0)
valid_ids = df_train_meta.index.intersection(df_train_latent.index).intersection(df_train.index)

df_train_meta = df_train_meta.loc[valid_ids]
df_train = df_train.loc[valid_ids]
df_train_latent = df_train_latent.loc[valid_ids]

# Load training-time batch labels and UMAP/KMeans models (so downstream
# OLS batch correction matches what the trained GAE actually saw).
saved_umap = None
saved_kmeans = None

if not is_sensitivity:
    import joblib
    saved_labels = pd.read_csv(
        'batch_artifacts/inferred_draw_year_traintime.csv', index_col=0
    )
    saved_umap = joblib.load('batch_artifacts/umap_model_traintime.joblib')
    saved_kmeans = joblib.load('batch_artifacts/kmeans_model_traintime.joblib')
    
    df_train_meta['inferred_draw_year'] = saved_labels.loc[
        df_train_meta.index, 'inferred_draw_year'
    ]
    assert df_train_meta['inferred_draw_year'].notna().all(), \
        "Training-time labels missing for some discovery patients"
    print(f"Loaded train-time batch labels: "
          f"{df_train_meta['inferred_draw_year'].value_counts().to_dict()}")

train_latent_raw = df_train_latent.values.astype(np.float64)

# save the metadata file which can then be used in control extraction scripts.
df_train_meta.to_csv('discovery_pathway_input_metadata_AGEMATCHED.csv', index=False)

# =========================================================
# 3. LOAD VALIDATION DATA & PREDICT
# =========================================================
print("\n=== STEP 3: Load Validation Data ===")
df_val = pd.read_parquet(VAL_PARQUET)
df_val_meta = pd.read_csv(VAL_METADATA).set_index(ID_COL)

# align protein columns to match that of tge discovery
# STRICT ALIGNMENT: Force Validation to perfectly match the pre-trained Discovery network
missing_cols = set(df_train.columns) - set(df_val.columns)
if missing_cols:
    print(f"Warning: Validation is missing {len(missing_cols)} proteins. Filling with 0 (mean expression).")
# Reindex forces df_val to have the exact same columns in the exact same order as df_train
df_val = df_val.reindex(columns=df_train.columns, fill_value=0.0)

# Filter by MMSE for early-stage AD (15 <= MMSE <= 30)
df_val_meta[COGNITIVE_COL] = pd.to_numeric(df_val_meta[COGNITIVE_COL], errors='coerce')
mask_test_valid = (df_val_meta[COGNITIVE_COL] >= 15) & (df_val_meta[COGNITIVE_COL] <= 30)

# <--- NEW: STRICT LATE-ONSET FILTER FOR VALIDATION --->
df_val_meta[AGE_COL] = pd.to_numeric(df_val_meta[AGE_COL], errors='coerce')
mask_test_load = df_val_meta[AGE_COL] >= 65.0

# Calculate sample sizes to see the effect
original_val_n = mask_test_valid.sum()
mask_test_valid = mask_test_valid & mask_test_load
final_val_n = mask_test_valid.sum()

print(f"Validation Cohort size before age filter: {original_val_n}")
print(f"Validation Cohort size after age >= 65 filter: {final_val_n}")
print(f"-> Dropped {original_val_n - final_val_n} early-onset patients from Validation (Multi-Site).")

df_val_meta = df_val_meta[mask_test_valid]

common_ids = df_val.index.intersection(df_val_meta.index)
df_val = df_val.loc[common_ids]
df_val_meta = df_val_meta.loc[common_ids]

# Scale Validation Inputs
X_val_scaled = np.zeros_like(df_val.values, dtype=np.float32)

if not is_sensitivity:
    # Validation cohort is multi-site -> scale per contributor
    contributors = df_val_meta[CONTRIB_COL].values
    for site in np.unique(contributors):
        idx = np.where(contributors == site)[0]
        X_val_scaled[idx] = StandardScaler().fit_transform(df_val.values[idx])
else:
    # Validation is Cohort F -> infer batches (draw year)
    saved_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    embedding = saved_umap.fit_transform(df_val.values) 
    saved_kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    df_val_meta[VAL_BATCH_COL] = saved_kmeans.fit_predict(embedding)
    
    batches = df_val_meta[VAL_BATCH_COL].values
    for b_id in np.unique(batches):
        idx = np.where(batches == b_id)[0]
        X_val_scaled[idx] = StandardScaler().fit_transform(df_val.values[idx])

print("Pre-computing Graph Convolutions for Validation Projection...")
adj_matrix = np.load(ADJACENCY_FILE).astype(np.float32)
X_val_smoothed = X_val_scaled @ adj_matrix

X_val_reshaped = np.expand_dims(X_val_smoothed, axis=-1).astype(np.float32)
num_proteins = X_val_reshaped.shape[1]

# Extract Model Parameters dynamically from the best_model_folder string
# e.g., "Exp_Lat32_L20.0001_Drp0.2_LR0.0005_Bch32_Fnl512"
best_lat = int(best_model_folder.split('_Lat')[1].split('_')[0])
best_drop = float(best_model_folder.split('_Drp')[1].split('_')[0])
best_funnel = int(best_model_folder.split('_Fnl')[1].split('_')[0] if '_' in best_model_folder.split('_Fnl')[1] else best_model_folder.split('_Fnl')[1])

encoder = build_encoder(num_proteins, latent_dim=best_lat, funnel_dim=best_funnel, drop_rate=best_drop)
encoder.load_weights(ENCODER_WEIGHTS)
val_latent_raw = encoder.predict(X_val_reshaped).astype(np.float64)

pd.DataFrame(
    val_latent_raw,
    index=df_val.index,
    columns=[f'L{i}' for i in range(val_latent_raw.shape[1])]
).to_csv('val_projected_latent.csv')

# Standardize Latent Space
latent_scaler = StandardScaler()
train_latent_norm = latent_scaler.fit_transform(train_latent_raw)
val_latent_norm = latent_scaler.transform(val_latent_raw)

# =========================================================
# 3.5. COHORT AGE COMPARISON
# =========================================================
print("\n=== STEP 3.5: Cohort Age Comparison ===")

# Safely extract and convert ages to numeric, dropping any NaNs
age_dis = pd.to_numeric(df_train_meta[AGE_COL], errors='coerce').dropna()
age_val = pd.to_numeric(df_val_meta[AGE_COL], errors='coerce').dropna()

print(f"Discovery Cohort Age:  Mean = {age_dis.mean():.2f} ± {age_dis.std():.2f} (N={len(age_dis)})")
print(f"Validation Cohort Age: Mean = {age_val.mean():.2f} ± {age_val.std():.2f} (N={len(age_val)})")

# Run Welch's t-test (assumes unequal variance)
t_stat, p_val = stats.ttest_ind(age_dis, age_val, equal_var=False)
print(f"Welch's t-test p-value: {p_val:.4e}")
if p_val < 0.05:
    print("-> WARNING: There is a statistically significant age difference between cohorts.")
else:
    print("-> The cohorts are not significantly different in age.")

# Generate an overlapping density plot
plt.figure(figsize=(8, 6))
sns.kdeplot(age_dis, label='Discovery (Cohort F)', fill=True, alpha=0.5, color='blue')
sns.kdeplot(age_val, label='Validation (Multi-Site)', fill=True, alpha=0.5, color='orange')
plt.title("Age Distribution: Discovery vs Validation Cohorts")
plt.xlabel("Age at Visit")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f"{image_folder}/cohort_age_comparison.png", dpi=300)
plt.close()
print(f"Saved age comparison plot to '{image_folder}/cohort_age_comparison.png'")

# save the metadata file which can then be used in control extraction scripts.
df_val_meta.to_csv('validation_pathway_input_metadata_AGEMATCHED.csv', index=False)

# =========================================================
# 4. K-MEANS CLUSTERING & OPTIMIZATION
# =========================================================
print("\n=== STEP 4: Clustering & Optimization ===")

# --- 4.1 Silhouette Score Checking ---
print("OPTIMIZATION: Checking optimal k (2-10)")
print(f"{'k':<5} {'Silhouette Score':<20}")
print("-" * 30)
    
# Save Silhouette/Elbow plots ---
silhouettes = []
inertias = []
k_range = range(2, 11)

for k in k_range:
    temp_kmeans = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
    labels = temp_kmeans.fit_predict(train_latent_norm)
    sil_score = silhouette_score(train_latent_norm, labels)
    silhouettes.append(sil_score)
    inertias.append(temp_kmeans.inertia_)
    print(f"{k:<5} {sil_score:<20.4f}")

# Generate the Plots
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow)', color='tab:blue')
ax1.plot(k_range, inertias, marker='o', color='tab:blue', label='Inertia')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  
ax2.set_ylabel('Silhouette Score', color='tab:red')
ax2.plot(k_range, silhouettes, marker='s', color='tab:red', label='Silhouette')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title("K-Means Optimization: Inertia and Silhouette Scores")
fig.tight_layout()
plt.savefig(f"{image_folder}/kmeans_optimization_k3_defense.png", dpi=300)
plt.close()
print(f"Saved K-optimization plot to '{image_folder}/kmeans_optimization_k3_defense.png'")

print("\n--- Generating Labels and Tracking Transitions (K=2 -> K=5) ---")
k_values_to_track = [2, 3, 4, 5]

for k in k_values_to_track:
    print(f"Fitting K-Means (k={k})...")
    kmeans_temp = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
    dis_labels = kmeans_temp.fit_predict(train_latent_norm)
    val_labels = kmeans_temp.predict(val_latent_norm)
    
    # Save all variations to Metadata so we can track the splits
    df_train_meta[f'Cluster_K{k}'] = dis_labels
    df_val_meta[f'Cluster_K{k}'] = val_labels

print("\n--- Crosstabs for K values (2 to 5) ---")
for k in k_values_to_track[:-1]:
    current_k = f'Cluster_K{k}'
    next_k    = f'Cluster_K{k+1}'
    print(f"\nTransition: {current_k}  -->  {next_k}")
    crosstab = pd.crosstab(df_train_meta[current_k], df_train_meta[next_k])
    print(crosstab)

# --- 4.3 Lock in Final Subtypes (K=3) ---
N_CLUSTERS = 3
col_with_final_labels = f'Cluster_K{N_CLUSTERS}'

# --- Demographic & Clinical Checks ---
print("\n=== Demographics & MMSE Across Clusters ===")
def check_demographics(df_meta, cluster_col, cohort_name):
    print(f"\n--- {cohort_name} ---")
    clusters = sorted(df_meta[cluster_col].dropna().unique())
    
    # 1. Continuous Variables (ANOVA)
    for col in [COGNITIVE_COL, AGE_COL]:
        if col in df_meta.columns:
            groups = [df_meta[df_meta[cluster_col] == c][col].dropna().values for c in clusters]
            if all(len(g) > 0 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                print(f"{col} ANOVA P-value: {p_val:.4f} " + ("(SIGNIFICANT DIFFERENCE!)" if p_val < 0.05 else "(Uniformly distributed - Good!)"))
                
    # 2. Categorical Variables (Chi-Square)
    for col in [SEX_COL, "apoe4_status"]: 
        if col in df_meta.columns:
            contingency = pd.crosstab(df_meta[cluster_col], df_meta[col])
            chi2, p_val, dof, ex = stats.chi2_contingency(contingency)
            print(f"{col} Chi-Square P-value: {p_val:.4f} " + ("(SIGNIFICANT SKEW!)" if p_val < 0.05 else "(Evenly distributed - Good!)"))

check_demographics(df_train_meta, col_with_final_labels, "Discovery Cohort")

# Check exact means and standard deviations
print("\n--- Discovery Cohort Clinical Means ---")
print(df_train_meta.groupby(col_with_final_labels)[[COGNITIVE_COL, AGE_COL]].agg(['mean', 'std', 'count']))

print("\n--- Discovery Cohort Sex Ratio (%) ---")
print(pd.crosstab(df_train_meta[col_with_final_labels], df_train_meta[SEX_COL], normalize='index') * 100)

print("\nChecking if MMSE is associated with final clusters - Discovery")
print(df_train_meta.groupby(col_with_final_labels)[COGNITIVE_COL].describe())
print("\nChecking if MMSE is associated with final clusters - Validation")
print(df_val_meta.groupby(col_with_final_labels)[COGNITIVE_COL].describe())

# --- 4.4 Independent Validation Clusters ---
print("\n--- Generating Independent De Novo Validation Clusters ---")
kmeans_val = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=RANDOM_STATE)
df_val_meta['Cluster_indep'] = kmeans_val.fit_predict(val_latent_norm)

print("\nAlignment: Projected Clusters vs Independent Clusters")
print(pd.crosstab(df_val_meta[col_with_final_labels], df_val_meta['Cluster_indep']))

# =========================================================
# 5. GENERATE PLOTS 
# =========================================================
print("\n=== STEP 5: Generating Plots ===")
plot_latent_alignment(train_latent_norm, val_latent_norm, image_folder, RANDOM_STATE)

generate_discovery_plots(train_latent_norm, df_train_meta, col_with_final_labels, COGNITIVE_COL, image_folder, RANDOM_STATE)

generate_validation_plots(val_latent_norm, df_val_meta, col_with_final_labels, CONTRIB_COL, VAL_BATCH_COL, COGNITIVE_COL, image_folder, RANDOM_STATE)

# =========================================================
# 6. MASTER PATHWAY ANALYSIS FUNCTION
# =========================================================
print("\nMapping Genes from Analyte File...")
analyte_df = pd.read_csv(ANALYTE_FILE, usecols=['apt_name', 'entrez_gene_symbol']).dropna(subset=['entrez_gene_symbol'])
analyte_df['entrez_gene_symbol'] = analyte_df['entrez_gene_symbol'].astype(str)
analyte_df['apt_name'] = analyte_df['apt_name'].str.replace('.', '_', regex=False)
GLOBAL_ID_TO_GENE = dict(zip(analyte_df['apt_name'], analyte_df['entrez_gene_symbol']))

def run_pathway_analysis(df_prot_disease, df_meta_disease, control_prot_file, control_meta_file, 
                         cluster_label_col, batch_col, output_dir, is_discovery=False,
                         umap_model=None, kmeans_model=None):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"RUNNING PATHWAY ANALYSIS -> {os.path.basename(output_dir)}")
    print(f"=======================================================")

    # 1. Load Controls
    df_prot_control = pd.read_parquet(control_prot_file)
    ctrl_df = pd.read_csv(control_meta_file)
    ctrl_df = ctrl_df.set_index('sample_id')
        
    common_ids_ctrl = df_prot_control.index.intersection(ctrl_df.index)
    df_prot_control = df_prot_control.loc[common_ids_ctrl]
    ctrl_df = ctrl_df.loc[common_ids_ctrl]
    print(f"size of controls is {ctrl_df.shape}")

    # 2. Align Control columns to Disease columns 
    missing_ctrl_cols = set(df_prot_disease.columns) - set(df_prot_control.columns)
    if missing_ctrl_cols:
        print(f"Warning: Controls missing {len(missing_ctrl_cols)} proteins. Filling with 0.")
    df_prot_control = df_prot_control.reindex(columns=df_prot_disease.columns, fill_value=0.0)

    # 3. Assign Batches to Controls (Now mathematically safe!)
    if batch_col == 'inferred_draw_year' and umap_model is not None and kmeans_model is not None:
        print("Projecting controls into the original UMAP space to assign batches...")
        emb_ctrl = umap_model.transform(df_prot_control.values)
        ctrl_df[batch_col] = kmeans_model.predict(emb_ctrl)
        print(ctrl_df[batch_col].value_counts())

    # 4. Merge Cases and Controls (Simplified because columns are already perfectly aligned)
    df_prot = pd.concat([df_prot_disease, df_prot_control], axis=0)

    ad_df = df_meta_disease.copy()
    if 'sample_id' in ad_df.columns:
        ad_df = ad_df.set_index('sample_id')

    ad_df['Group'] = 'Cluster ' + ad_df[cluster_label_col].astype(int).astype(str)
    ctrl_df['Group'] = 'Control'

    # Strip out any duplicate columns
    ad_df = ad_df.loc[:, ~ad_df.columns.duplicated()]
    ctrl_df = ctrl_df.loc[:, ~ctrl_df.columns.duplicated()]

    cols_keep = ['Group', batch_col, AGE_COL, SEX_COL]
    meta_combined = pd.concat([ad_df[cols_keep], ctrl_df[cols_keep]], axis=0)
    
    # Ensure no patients accidentally exist in BOTH Cases and Controls
    meta_combined = meta_combined[~meta_combined.index.duplicated(keep='first')]
    df_prot = df_prot[~df_prot.index.duplicated(keep='first')]
    
    common_ids = meta_combined.index.intersection(df_prot.index)
    df_prot = df_prot.loc[common_ids]
    meta_combined = meta_combined.loc[common_ids]

    background_genes = list(set([GLOBAL_ID_TO_GENE.get(x) for x in df_prot.columns if x in GLOBAL_ID_TO_GENE]))
    clusters = sorted([c for c in meta_combined['Group'].unique() if c != 'Control'])
    all_pathway_results = []

    print(meta_combined['Group'].value_counts())
    
    # 5. Run Robust OLS and GSEAPY
    for cluster in clusters:
        print(f"  -> Analyzing {cluster} vs Control...")
        group_cases = meta_combined[meta_combined['Group'] == cluster].index
        group_ctrls = meta_combined[meta_combined['Group'] == 'Control'].index
        
        comp_meta = meta_combined.loc[group_cases.union(group_ctrls)].copy()
        comp_data = df_prot.loc[comp_meta.index]
        
        comp_meta['Is_Disease'] = (comp_meta['Group'] == cluster).astype(int)
        comp_meta[AGE_COL] = pd.to_numeric(comp_meta[AGE_COL], errors='coerce').fillna(comp_meta[AGE_COL].mean())
        
        X = pd.DataFrame({'Intercept': 1, 'Is_Disease': comp_meta['Is_Disease'], 'Age': comp_meta[AGE_COL]}, index=comp_meta.index)
        
        batch_dummies = pd.get_dummies(comp_meta[batch_col], prefix='Batch', drop_first=True, dtype=int)
        sex_dummies = pd.get_dummies(comp_meta[SEX_COL], prefix='Sex', drop_first=True, dtype=int)
        X = pd.concat([X, batch_dummies, sex_dummies], axis=1).astype(float)
        
        Y = comp_data.values
        protein_names = comp_data.columns
        
        X_mat = X.values
        XtX_pinv = np.linalg.pinv(X_mat.T @ X_mat)
        betas = XtX_pinv @ X_mat.T @ Y  
        
        Y_pred = X_mat @ betas
        residuals = Y - Y_pred
        dof = X_mat.shape[0] - X_mat.shape[1]
        
        sigma_sq = np.sum(residuals**2, axis=0) / dof 
        target_idx = X.columns.get_loc('Is_Disease')
        
        var_beta = XtX_pinv[target_idx, target_idx] * sigma_sq
        se_beta = np.sqrt(var_beta)
        
        t_stats = betas[target_idx, :] / se_beta
        p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)
        
        res_df = pd.DataFrame({'Aptamer': protein_names, 'Gene': [GLOBAL_ID_TO_GENE.get(p, p) for p in protein_names], 'Log2FC': betas[target_idx, :], 'P': p_vals})
        res_df = res_df[res_df['Gene'] != res_df['Aptamer']].dropna(subset=['P'])
        res_df['FDR'] = multipletests(res_df['P'], method='fdr_bh')[1]
        res_df.to_csv(f"{output_dir}/DE_stats_{cluster.replace(' ', '_')}_vs_Control.csv", index=False)
        
        up_genes = res_df[(res_df['FDR'] < 0.05) & (res_df['Log2FC'] > 0.1)]['Gene'].unique().tolist()
        down_genes = res_df[(res_df['FDR'] < 0.05) & (res_df['Log2FC'] < -0.1)]['Gene'].unique().tolist()
        
        for direction, gene_list in [('UP', up_genes), ('DOWN', down_genes)]:
            if len(gene_list) < 5: continue
            try:
                enr = gp.enrich(gene_list=gene_list, gene_sets=['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022'], background=background_genes, outdir=None)
                sigs = enr.results[enr.results['Adjusted P-value'] < 0.05].copy()
                if not sigs.empty:
                    sigs['Cluster'] = cluster
                    sigs['Direction'] = direction
                    sigs['Comparison'] = f"{cluster} {direction}"
                    all_pathway_results.append(sigs)
            except Exception as e:
                pass

    if all_pathway_results:
        full_df = pd.concat(all_pathway_results)
        full_df.to_csv(f"{output_dir}/All_Clusters_Pathway_Enrichment.csv", index=False)
        for cluster in full_df['Cluster'].unique():
            cluster_df = full_df[full_df['Cluster'] == cluster]
            cluster_df.to_csv(f"{output_dir}/{cluster.replace(' ', '_')}_Pathways.csv", index=False)

        plot_df = pd.DataFrame()
        for comp in full_df['Comparison'].unique():
            top = full_df[full_df['Comparison'] == comp].sort_values('Adjusted P-value').head(15)
            plot_df = pd.concat([plot_df, top])
            
        plot_df['Term_Clean'] = plot_df['Term'].apply(lambda x: x.split(' (GO:')[0])
        plot_df['Significance'] = -np.log10(plot_df['Adjusted P-value'])
        plot_df['Count'] = plot_df.apply(lambda row: int(str(row['Overlap']).split('/')[0]) if 'Overlap' in row else 5, axis=1)

        plt.figure(figsize=(12, 14))
        sns.set_style("whitegrid")
        plot_df = plot_df.sort_values(['Cluster', 'Direction', 'Significance'])
        sns.scatterplot(data=plot_df, x='Comparison', y='Term_Clean', size='Count', sizes=(20, 300), hue='Significance', palette='viridis', edgecolor='black')
        plt.title(f"Pathway Enrichment ({os.path.basename(output_dir)})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="-Log10(FDR)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Cluster_Pathway_DotPlot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Enrichment complete and plotted.")
    else:
        print("No significant pathways found.")

df_val_meta.to_csv('all_clusters_validation.csv', index=True)
df_train_meta.to_csv('all_clusters_discovery.csv', index=True)

# =========================================================
# 7. EXECUTE ALL PATHWAY RUNS
# =========================================================

# 1. Discovery Run
run_pathway_analysis(
    df_prot_disease=df_train,
    df_meta_disease=df_train_meta,
    control_prot_file=DIS_CTRL_PROT,
    control_meta_file=DIS_CTRL_META,
    cluster_label_col=col_with_final_labels,
    batch_col=DIS_BATCH_COL,
    output_dir=OUT_DIS_PATHWAYS,
    is_discovery=True,
    umap_model=saved_umap if not is_sensitivity else None,   
    kmeans_model=saved_kmeans if not is_sensitivity else None
)

# 2. Validation Run (Projected Clusters)
run_pathway_analysis(
    df_prot_disease=df_val,
    df_meta_disease=df_val_meta,
    control_prot_file=VAL_CTRL_PROT,
    control_meta_file=VAL_CTRL_META,
    cluster_label_col=col_with_final_labels,
    batch_col=VAL_BATCH_COL,
    output_dir=OUT_VAL_PROJ_PATHWAYS,
    is_discovery=False,
    umap_model=saved_umap if is_sensitivity else None,   
    kmeans_model=saved_kmeans if is_sensitivity else None
)

# 3. Validation Run (Independent De Novo Clusters)
run_pathway_analysis(
    df_prot_disease=df_val,
    df_meta_disease=df_val_meta,
    control_prot_file=VAL_CTRL_PROT,
    control_meta_file=VAL_CTRL_META,
    cluster_label_col='Cluster_indep',
    batch_col=VAL_BATCH_COL,
    output_dir=OUT_VAL_INDEP_PATHWAYS,
    is_discovery=False,
    umap_model=saved_umap if is_sensitivity else None,     
    kmeans_model=saved_kmeans if is_sensitivity else None
)

print("\nPipeline Complete!")
