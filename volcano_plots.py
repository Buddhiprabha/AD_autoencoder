'''
This script creates the volcano plots for ELS, ELA, and CCMS.

SET the following variables before running.

- DISC_PATHWAY_DIR: this matches the output folder from main_pipeline.py script for
                    the discovery cohort
- is_independent: set to True if you want to compare against the de-novo independent
                  validation, to False for the projected validation.
- VAL_PATHWAY_DIR: pathways_val_projected or pathways_val_independent from main_pipeline.py
- OUT_DIR_CSV: to save cluster exclusive proteins from this script as CSV
- OUTPUT_DIR: to save volcano plots produced from this script
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text
import os

is_independent = True

# =========================================================
# 1. CONFIGURATION
# =========================================================

# --- Files (Discovery) ---
DISC_PATHWAY_DIR  = "pathways_discovery"

DISC_PATHWAY_FILE = f"{DISC_PATHWAY_DIR}/All_Clusters_Pathway_Enrichment.csv"
OUT_DIR_CSV = "exclusive_pathways_genes"
os.makedirs(OUT_DIR_CSV, exist_ok=True)

if not is_independent:
    VAL_PATHWAY_DIR   = "pathways_val_projected"
    OUTPUT_DIR = "volcano_plots_projected"
else:
    VAL_PATHWAY_DIR   = "pathways_val_independent"
    OUTPUT_DIR = "volcano_plots_independent"
    
VAL_PATHWAY_FILE  = f"{VAL_PATHWAY_DIR}/All_Clusters_Pathway_Enrichment.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Plot Settings ---
P_THRESH = 0.05        # FDR Significance
FC_THRESH = 0.1

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def get_pathway_genes_by_cluster(pathway_csv):
    if not os.path.exists(pathway_csv): return {}
    df = pd.read_csv(pathway_csv)
    out = {}

    for cl, grp in df.groupby('Cluster'):
        gene_counts = {}
        for genes_str in grp['Genes'].dropna():
            for g in str(genes_str).split(';'):
                g = g.strip()
                if g: gene_counts[g] = gene_counts.get(g, 0) + 1
        out[str(cl)] = gene_counts
    return out

# =========================================================
# 3. VOLCANO PLOTTER (Using Pre-Calculated Stats)
# =========================================================
def draw_volcano(stats_dir, cohort_name, pathway_genes_dict):
    print(f"\n--- Generating Ranked Volcano for {cohort_name} ---")
    x_label = "Log2 Fold Change" 
    
    # Find all the pre-calculated stats CSVs in the directory
    stats_files = [f for f in os.listdir(stats_dir) if f.startswith('DE_stats_') and f.endswith('.csv')]
    
    for f in stats_files:
        # extract cluster name from the file name (file names are like DE_stats_Cluster_0_vs_Control.csv)
        cluster = f.replace('DE_stats_', '').replace('_vs_Control.csv', '').replace('_', ' ')
        print(f"  Plotting {cluster} vs Control...")
        
        # Load the pre-calculated DE data
        res_df = pd.read_csv(os.path.join(stats_dir, f))
        res_df['Gene'] = res_df['Gene'].astype(str).str.strip()
        
        # Ensure correct formatting for plotting
        # res_df['-Log10(P)'] = res_df['P'].apply(lambda x: -np.log10(x) if x > 0 else 300)
        # Ensure correct formatting for plotting (Dynamic Ceiling)
        min_p = res_df.loc[res_df['P'] > 0, 'P'].min()
        dynamic_ceiling = -np.log10(min_p) + 2  # Place 0.0 values 2 units above the top real point
        res_df['-Log10(P)'] = res_df['P'].apply(lambda x: -np.log10(x) if x > 0 else dynamic_ceiling)
        
        # define the cluster key in pathway files (these are saved as Cluster 0, Cluster 1, and Cluster 2)
        cluster_key = str(cluster).replace('_', ' ')
        # cluster_key = str(cluster).replace(' ', '_')
        # print(pathway_genes_dict)
        
        # get pathway gene counts for the current cluster
        cluster_gene_counts = pathway_genes_dict.get(cluster_key, {})
        
        # add new columns to the DE data frame that are used in plotting
        # add how many different pathways the gene appears (using pandas map function)
        res_df['Pathway_Count'] = res_df['Gene'].map(cluster_gene_counts).fillna(0).astype(int)
        # calculate a rank score for each protein which is used in selecting which genes to label on the volcano plot
        res_df['Rank_Score'] = res_df['-Log10(P)'] * res_df['Log2FC'].abs()
        # add a new column that specifies whether a protein a significant based on FDR and log2FC size
        res_df['Sig'] = (res_df['FDR'] < P_THRESH) & (res_df['Log2FC'].abs() > FC_THRESH)
        
        # define conditions for label colouring
        conditions = [(res_df['Sig'] & (res_df['Log2FC'] > 0)), (res_df['Sig'] & (res_df['Log2FC'] < 0))]
        # add conditional label colours to the data frame
        res_df['DotColor'] = np.select(conditions, ['#D32F2F', '#1976D2'], default='lightgrey')
        
        plt.figure(figsize=(10, 9))
        
        # Plot Grey dots first (Background layer with zorder=1)
        mask_grey = res_df['DotColor'] == 'lightgrey'
        plt.scatter(res_df.loc[mask_grey, 'Log2FC'], res_df.loc[mask_grey, '-Log10(P)'], 
                    c='lightgrey', s=40, alpha=0.5, edgecolors='white', linewidth=0.3, zorder=1)
        
        # Plot Colored dots ON TOP (Foreground layer)
        mask_color = ~mask_grey
        plt.scatter(res_df.loc[mask_color, 'Log2FC'], res_df.loc[mask_color, '-Log10(P)'], 
                    c=res_df.loc[mask_color, 'DotColor'], s=55, alpha=0.9, edgecolors='white', linewidth=0.5, zorder=5)

        # draw thresholds in black (k) dotted (:) lines
        plt.axvline(x=FC_THRESH, color='k', ls=':', alpha=0.5); plt.axvline(x=-FC_THRESH, color='k', ls=':', alpha=0.5)
        plt.axhline(y=-np.log10(P_THRESH), color='k', ls=':', alpha=0.5)

        sig_rows = res_df[res_df['Sig']]
        top_hubs = sig_rows.sort_values(by=['Pathway_Count', '-Log10(P)'], ascending=False).head(15)
        top_up = sig_rows[(sig_rows['Log2FC'] > 0)].sort_values(by='Rank_Score', ascending=False).head(20)
        top_down = sig_rows[(sig_rows['Log2FC'] < 0)].sort_values(by='Rank_Score', ascending=False).head(20)
        to_label = pd.concat([top_hubs, top_up, top_down]).sort_values('Rank_Score', ascending=False).drop_duplicates(subset=['Gene'])

        texts = []
        for _, row in to_label.iterrows():
            fw = 'bold' if row['Pathway_Count'] >= 10 else 'normal'
            fc = 'purple' if row['Pathway_Count'] == 0 else 'black'
            texts.append(plt.text(row['Log2FC'], row['-Log10(P)'], row['Gene'], fontsize=14, fontweight=fw, color=fc, zorder=10))
            
        if texts: adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), force_text=0.5, expand_points=(3.5, 3.5))

        clean_name = str(cluster).replace(" ", "_")
        plt.title(f"{cohort_name}: {cluster} vs Control\n(OLS Age/Sex Corrected. Bold = 10+ Pathways)", fontsize=14)
        plt.xlabel(x_label, fontsize=14) 
        plt.ylabel("-Log10 P-value", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/Volcano_{cohort_name}_{clean_name}.png", dpi=300)
        plt.close()

# =========================================================
# 4. MAIN EXECUTION
# =========================================================

path_genes_disc = get_pathway_genes_by_cluster(DISC_PATHWAY_FILE)
draw_volcano(DISC_PATHWAY_DIR, "Discovery", path_genes_disc)

path_genes_val  = get_pathway_genes_by_cluster(VAL_PATHWAY_FILE)
draw_volcano(VAL_PATHWAY_DIR, "Validation", path_genes_val)

# =========================================================
# 5. GENERATE REPLICATION-ONLY PLOTS (Using Pre-Calculated Stats)
# =========================================================
print("\n--- Generating Replicated-Only Discovery Plots ---")
# try:
    # find all proteins in each cluster
pathway_map = get_pathway_genes_by_cluster(DISC_PATHWAY_FILE)

print("\n--- DIAGNOSTIC CHECK ---")
print(f"Total Clusters Loaded: {len(pathway_map)}")

if pathway_map:
    sample_cluster = list(pathway_map.keys())[0]
    print(f"Sample Pathway Genes (from {sample_cluster}):", list(pathway_map[sample_cluster].keys())[:5])
else:
    print("WARNING: pathway_map is EMPTY. Check your file path!")

# Load one DE stats file just to peek at the genes
test_d_file = f"{DISC_PATHWAY_DIR}/DE_stats_Cluster_0_vs_Control.csv"
if os.path.exists(test_d_file):
    test_res_d = pd.read_csv(test_d_file)
    print("Sample DE Genes:", test_res_d['Gene'].dropna().head().tolist())
print("------------------------\n")

if not is_independent:
    cluster_map = {'Cluster 0': 'Cluster 0', 'Cluster 1': 'Cluster 1', 'Cluster 2': 'Cluster 2'}
else:
    cluster_map = {'Cluster 0': 'Cluster 2', 'Cluster 1': 'Cluster 1', 'Cluster 2': 'Cluster 0'}
    
clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    
for c_disc in clusters:
    c_val = cluster_map.get(c_disc)
    print(f"  Matching {c_disc} (Dis) <-> {c_val} (Val)")
        
    # Load Discovery Stats from file
    d_file = f"{DISC_PATHWAY_DIR}/DE_stats_{c_disc.replace(' ', '_')}_vs_Control.csv"
    if not os.path.exists(d_file): continue
    res_d = pd.read_csv(d_file)
    res_d['Gene'] = res_d['Gene'].astype(str).str.strip()
    res_d = res_d.sort_values('P').drop_duplicates(subset=['Gene'])
    
    # res_d['-Log10(P)'] = res_d['P'].apply(lambda x: -np.log10(x) if x > 0 else 300)
    # Dynamic Ceiling for Replication plot
    min_p_d = res_d.loc[res_d['P'] > 0, 'P'].min()
    dynamic_ceiling_d = -np.log10(min_p_d) + 2
    res_d['-Log10(P)'] = res_d['P'].apply(lambda x: -np.log10(x) if x > 0 else dynamic_ceiling_d)
        
    # ============ get the proteins exclusive to the current cluster (DIRECTIONAL LOGIC) ====================
        
    c_path_genes = set(pathway_map.get(c_disc, {}).keys())
        # if not c_path_genes: c_path_genes = pathway_map.get(c_disc.replace(' ', '_'), set())
        
    all_pathway_genes = set() # Track every AD pathway gene
    for clus_name, clus_genes in pathway_map.items():
        all_pathway_genes.update(clus_genes.keys())
        
    # 1. Target cluster significant up and down genes
    c_sig_d = res_d[(res_d['FDR'] < P_THRESH) & (res_d['Log2FC'].abs() > FC_THRESH)]
    c_up_d = set(c_sig_d[c_sig_d['Log2FC'] > 0]['Gene'])
    c_down_d = set(c_sig_d[c_sig_d['Log2FC'] < 0]['Gene'])

    # 2. Find what is UP and DOWN in the other clusters
    other_up_genes = set()
    other_down_genes = set()
    for other_c in clusters:
        if other_c != c_disc:
            other_file = f"{DISC_PATHWAY_DIR}/DE_stats_{other_c.replace(' ', '_')}_vs_Control.csv"
            if not os.path.exists(other_file): continue
            tmp_df = pd.read_csv(other_file)
            sig_other = tmp_df[(tmp_df['FDR'] < P_THRESH) & (tmp_df['Log2FC'].abs() > FC_THRESH)]
            
            other_up_genes.update(sig_other[sig_other['Log2FC'] > 0]['Gene'])
            other_down_genes.update(sig_other[sig_other['Log2FC'] < 0]['Gene'])
    
    # 3. STATISTICAL Exclusivity (Defines the target set for Validation)
    pure_exclusive_up = c_up_d - other_up_genes
    pure_exclusive_down = c_down_d - other_down_genes
    
    # This is the exact set of genes that gets carried forward to validation
    all_stat_exclusive = pure_exclusive_up.union(pure_exclusive_down)

    # 4. PATHWAY Exclusivity (Used STRICTLY for coloring the Volcano Plot red/blue)
    pathway_exclusive_up = pure_exclusive_up.intersection(c_path_genes)
    pathway_exclusive_down = pure_exclusive_down.intersection(c_path_genes)
    directional_exclusive_path_genes = pathway_exclusive_up.union(pathway_exclusive_down)

    # =========== TARGETED FDR REPLICATION =============
    v_file = f"{VAL_PATHWAY_DIR}/DE_stats_{c_val.replace(' ', '_')}_vs_Control.csv"
    if not os.path.exists(v_file): continue
    res_v = pd.read_csv(v_file)
    res_v['Gene'] = res_v['Gene'].astype(str).str.strip()
    res_v = res_v.sort_values('P').drop_duplicates(subset=['Gene'])
        
    # The crucial step: filter the validation cohort to ONLY the pure exclusive genes
    res_v_targeted = res_v[res_v['Gene'].isin(all_stat_exclusive)].copy()
    
    replicated_genes = set()
    if not res_v_targeted.empty:
        # Recompute FDR on this restricted subset
        res_v_targeted['FDR_targeted'] = multipletests(res_v_targeted['P'], method='fdr_bh')[1]
        
        # Filter for significance
        sig_v_final = res_v_targeted[res_v_targeted['FDR_targeted'] < P_THRESH]
        
        # Merge with Discovery stats to ensure fold-change direction matches
        merged = pd.merge(res_d, sig_v_final, on='Gene', suffixes=('_d', '_v'))
        replicated_genes = set(merged[merged['Log2FC_d'] * merged['Log2FC_v'] > 0]['Gene'])
        
    # add new columns
    res_d['DotColor'] = 'lightgrey'
    
    # Save our boolean masks for plotting and accurate CSV exporting
    res_d['IsReplicated'] = res_d['Gene'].isin(replicated_genes)
    res_d['IsStatExclusive'] = res_d['Gene'].isin(all_stat_exclusive)          # Use this for CSV export
    res_d['IsPathwayExclusive'] = res_d['Gene'].isin(directional_exclusive_path_genes) # Use this for plot colors
    res_d['IsNovel'] = (~res_d['Gene'].isin(all_pathway_genes)) & res_d['IsStatExclusive'] # Novel AND exclusive
    
    # Make general replicated genes visible (Dark Grey) before coloring the special ones
    mask_rep_shared = (res_d['IsReplicated']) & (~res_d['IsPathwayExclusive']) & (~res_d['IsNovel'])
    res_d.loc[mask_rep_shared, 'DotColor'] = '#A0A0A0'
        
    # find replicated up-regulated exclusive proteins
    mask_rep_up_excl = (res_d['IsReplicated']) & (res_d['Log2FC'] > 0) & (res_d['IsPathwayExclusive'])
    res_d.loc[mask_rep_up_excl, 'DotColor'] = '#D32F2F' # red
        
    # find replicated down-regulated exclusive proteins
    mask_rep_down_excl = (res_d['IsReplicated']) & (res_d['Log2FC'] < 0) & (res_d['IsPathwayExclusive'])
    res_d.loc[mask_rep_down_excl, 'DotColor'] = '#1976D2' # blue
        
    # find replicated proteins that are not in pathways (potential novel biomarkers)
    mask_rep_novel = (res_d['IsReplicated']) & (res_d['IsNovel'])
    res_d.loc[mask_rep_novel, 'DotColor'] = 'purple'

    plt.figure(figsize=(10, 9))
        
    mask_grey = res_d['DotColor'] == 'lightgrey'
    plt.scatter(res_d.loc[mask_grey, 'Log2FC'], res_d.loc[mask_grey, '-Log10(P)'], 
                c='lightgrey', s=40, alpha=0.5, edgecolors='white', linewidth=0.3, zorder=1)
        
    mask_color = ~mask_grey
    plt.scatter(res_d.loc[mask_color, 'Log2FC'], res_d.loc[mask_color, '-Log10(P)'], 
                c=res_d.loc[mask_color, 'DotColor'], s=55, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=5)

    plt.axvline(x=FC_THRESH, color='k', ls=':', alpha=0.5); plt.axvline(x=-FC_THRESH, color='k', ls=':', alpha=0.5)
    plt.axhline(y=-np.log10(P_THRESH), color='k', ls=':', alpha=0.5)
        
    label_candidates = res_d[(res_d['IsReplicated']) & (res_d['IsPathwayExclusive'] | res_d['IsNovel'])]
    reps_up = label_candidates[label_candidates['Log2FC'] > 0].sort_values('P')
    reps_down = label_candidates[label_candidates['Log2FC'] < 0].sort_values('P')
        
    top_up = reps_up.head(10)
    top_down = reps_down.head(10)
    # remaining = label_candidates.drop(top_up.index).drop(top_down.index, errors='ignore')
    remaining = label_candidates.drop(top_up.index).drop(top_down.index)
    top_rest = remaining.sort_values('P').head(15)
    
    specific_genes = []
    
    # Grab these specific genes from the replicated dataset
    must_label = res_d[(res_d['Gene'].isin(specific_genes)) & (res_d['IsReplicated'])]
    
    # to_label = pd.concat([top_up, top_down, top_rest])
    # --- manuscript-highlighted proteins to force-label (added to the auto-selected genes) ---
    # These are the proteins named in the manuscript text, per cluster. They are labelled in
    # ADDITION to the auto-selected top genes. the drop_duplicates below guarantees a protein
    # passing both criteria is labelled only once.
    MANUSCRIPT_GENES = {
        'Cluster 1': [  # ELA
            'UBA1','UBA6','EIF4B','EIF4E','CHMP1B','IST1','VTA1','RAB5B','DNM2','ATP6V1C2',  # up-exclusive
            'EPHB2','NRCAM',                                                                  # down-exclusive
        ],
        'Cluster 0': [  # ELS
            'SMOC1','NCMAP','CD3E','PSMA5','EDIL3','CXCL12','SEMA6A','UNC5B',                 # up-exclusive
            'CAPZA1','ARPC1B','EIF4G1','SNX5','ATP6V1C2',                                     # down-exclusive
        ],
        'Cluster 2': [  # CCMS
            'F2','MASP1','APOB','BMP6','IL2', 'C3',                                         # up (gene-level complement/coag)
            'CCT5','ENO1','IDH1','PGD','APRT','CTPS1','IDI1','SHMT2','EIF1B','GLRX3','FLT3',  # down (metabolic/proteostatic)
        ],
    }
    manuscript_genes = MANUSCRIPT_GENES.get(str(c_disc), [])
    # lowest-P aptamer per manuscript gene; NOT gated on the script's IsReplicated flag, so the
    # plot always shows the paper's named proteins.
    manuscript_hits = (res_d[res_d['Gene'].isin(manuscript_genes)]
                       .sort_values('P').drop_duplicates(subset=['Gene']))
    missing = sorted(set(manuscript_genes) - set(manuscript_hits['Gene']))
    if missing:
        print(f"  [warn] manuscript genes not found in {c_disc} DE table (not labelled): {missing}")

    # auto-selected + ad-hoc (specific_genes) + manuscript, de-duplicated by gene
    to_label = pd.concat([top_up, top_down, top_rest, must_label, manuscript_hits]).drop_duplicates(subset=['Gene'])
    
    print(res_d.loc[res_d['Gene']=='C3', ['Log2FC','P','IsReplicated','IsStatExclusive','DotColor']])
    
    print('C3 in res_d:', 'C3' in set(res_d['Gene']))
    print('C3 in to_label:', 'C3' in set(to_label['Gene']))
        
    texts = []
    for _, row in to_label.iterrows():
        is_manuscript = row['Gene'] in manuscript_genes
        if not is_manuscript and (row['DotColor'] == "A0A0A0" or not row['IsReplicated']):
            continue   # existing skip logic, but never skip manuscript genes
        lbl_color = 'purple' if row['IsNovel'] else 'black'
        # lbl_weight = 'bold' if row['IsNovel'] else 'normal'
        lbl_weight = 'bold'
        texts.append(plt.text(row['Log2FC'], row['-Log10(P)'], row['Gene'], fontsize=14, fontweight=lbl_weight, color=lbl_color, zorder=10))
        
    if texts: 
            adjust_text(texts, 
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7), 
                        force_text=3,       # Pushes text labels away from each other much harder
                        force_points=2.5,     # Pushes text away from the data dots
                        max_iter=4000)        # Gives the algorithm more time to untangle the mess
        
    cl_clean = str(c_disc).replace(" ", "_")
    plt.title(f"{c_disc} (Discovery) - Replicated Hits Only (OLS Corrected)\n(Red/Blue=Exclusive Pathway, Purple=Novel, Dark Grey=Other Replicated)", fontsize=14)
    plt.xlabel("Log2 Fold Change", fontsize=14)
    plt.ylabel("-Log10 P-value", fontsize=14)
    # plt.xlim(-1.6, 1.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Volcano_Replicated_{cl_clean}.png", dpi=300)
    plt.close()
    
    # Output the resulting annotated DataFrame to a CSV so we can analyze the labels
    csv_output_file = f"{OUTPUT_DIR}/Annotated_Replication_Stats_{cl_clean}.csv"
    res_d.to_csv(csv_output_file, index=False)
    print(f"  --> Saved annotated stats to: {csv_output_file}")
    
    # =========================================================
    # EXPORT FINAL REPLICATED EXCLUSIVE GENES TO DEDICATED FOLDER
    # =========================================================

    # Filter for genes that are Replicated, Exclusive, and split by Up/Down
    df_exc_up = res_d[(res_d['IsReplicated']) & (res_d['IsStatExclusive']) & (res_d['Log2FC'] > 0)].copy()
    df_exc_down = res_d[(res_d['IsReplicated']) & (res_d['IsStatExclusive']) & (res_d['Log2FC'] < 0)].copy()

    # Sort them by Log2FC magnitude for easier reading
    df_exc_up = df_exc_up.sort_values(by='Log2FC', ascending=False)
    df_exc_down = df_exc_down.sort_values(by='Log2FC', ascending=True)

    # Save to the new directory
    if not df_exc_up.empty:
        up_file = f"{OUT_DIR_CSV}/{cl_clean}_Replicated_Exclusive_UP.csv"
        df_exc_up.to_csv(up_file, index=False)
        print(f"  --> Saved {len(df_exc_up)} UP genes to: {up_file}")
        
    if not df_exc_down.empty:
        down_file = f"{OUT_DIR_CSV}/{cl_clean}_Replicated_Exclusive_DOWN.csv"
        df_exc_down.to_csv(down_file, index=False)
        print(f"  --> Saved {len(df_exc_down)} DOWN genes to: {down_file}")
        
