'''
This script has helper functions used by main_pipeline.py
'''

import umap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_latent_alignment(train_latent_norm, val_latent_norm, image_folder, random_state):
    print("\nGenerating UMAP for train + validation latent (replication)...")
    latent_combined = np.vstack([train_latent_norm, val_latent_norm])
    dataset_label   = (['Training'] * train_latent_norm.shape[0] +
                       ['Validation'] * val_latent_norm.shape[0])

    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(latent_combined)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], hue=dataset_label,
        palette={'Training': 'grey', 'Validation': 'red'}, alpha=0.5, s=15
    )
    plt.title("Latent space alignment (Training vs Validation)")
    plt.savefig(f'{image_folder}/latent_alignment_train_vs_val.png')
    plt.close()

def generate_discovery_plots(train_latent_norm, df_train_meta, col_with_final_labels, cognitive_col, image_folder, random_state):
    reducer_batch = umap.UMAP(random_state=random_state)
    emb_train_batch = reducer_batch.fit_transform(train_latent_norm)
    
    # PLOT 1: Cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta[col_with_final_labels].astype(str), alpha=0.7, s=15)
    plt.title("Discovery latent space coloured by cluster")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{image_folder}/discovery_latent_umap_by_cluster.png')
    plt.close()

    # PLOT 2: Comorbidities (Safe Check)
    disease_cols_raw = ['pd', 'ftd', 'als', 'ad']
    disease_cols = [col for col in disease_cols_raw if col in df_train_meta.columns]
    if len(disease_cols) > 0:
        def get_subtype(row):
            dis = ""
            for disease in disease_cols:
                if row[disease] == 1:
                    dis = dis + "_" + disease.upper() if dis != "" else disease.upper()
            return 'AD_only' if dis == "" else dis
            
        df_train_meta['subtype'] = df_train_meta.apply(get_subtype, axis=1)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta['subtype'], palette='Set1', alpha=0.7)
        plt.title("Discovery latent space coloured by disease")
        plt.tight_layout()
        plt.savefig(f'{image_folder}/discovery_latent_by_subtype_and_disease.png')
        plt.close()

    # PLOT 3: Age and Sex
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta['age_at_visit'], palette='turbo', alpha=0.7)
    plt.title("Discovery: Latent (coloured by age)")
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta['sex'], alpha=0.7)
    plt.title("Discovery: Latent (coloured by sex)")
    plt.tight_layout()
    plt.savefig(f'{image_folder}/discovery_latent_by_age_and_sex.png')
    plt.close()

    # PLOT 4: CDR
    if 'cdr' in df_train_meta.columns:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta[col_with_final_labels], palette='Set1', alpha=0.7)
        plt.title("Discovery: Latent (coloured by subtype)")
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta['cdr'], palette='Set1', alpha=0.7)
        plt.title("Discovery: Latent (coloured by cdr)")
        plt.tight_layout()
        plt.savefig(f'{image_folder}/discovery_latent_by_subtype_and_cdr.png')
        plt.close()

    # PLOT 5: Binned MMSE
    if cognitive_col in df_train_meta.columns:
        def bin_mmse(score):
            if score < 20: return '<20'
            elif 20 <= score <= 25: return '20-25'
            else: return '>25'
        df_train_meta['mmse_binned'] = df_train_meta[cognitive_col].apply(bin_mmse)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta[col_with_final_labels], palette='Set1', alpha=0.7)
        plt.title("Discovery: Latent (coloured by subtype)")
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=emb_train_batch[:, 0], y=emb_train_batch[:, 1], hue=df_train_meta['mmse_binned'], palette='viridis', alpha=0.7)
        plt.title("Discovery: Latent (coloured by mmse_binned)")
        plt.tight_layout()
        plt.savefig(f'{image_folder}/discovery_latent_by_subtype_and_mmse_binned.png')
        plt.close()

def generate_validation_plots(val_latent_norm, df_val_meta, col_with_final_labels, contrib_col, batch_col, cognitive_col, image_folder, random_state):
    reducer_val = umap.UMAP(random_state=random_state)
    emb_val = reducer_val.fit_transform(val_latent_norm)

    # PLOT 1: Contributor / Subtype
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_hue = contrib_col if contrib_col in df_val_meta.columns else batch_col
    sns.scatterplot(x=emb_val[:, 0], y=emb_val[:, 1], hue=df_val_meta[plot_hue] if plot_hue in df_val_meta.columns else None, alpha=0.7)
    plt.title("Validation latent (coloured by contributor)")
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=emb_val[:, 0], y=emb_val[:, 1], hue=df_val_meta[col_with_final_labels], alpha=0.7)
    plt.title("Validation latent (coloured by replicated subtype)")
    plt.tight_layout()
    plt.savefig(f'{image_folder}/val_latent_by_contributor_and_subtype.png')
    plt.close()

    # PLOT 2: Comorbidities
    disease_cols_raw = ['pd', 'ftd', 'als']
    disease_cols = [col for col in disease_cols_raw if col in df_val_meta.columns]
    if len(disease_cols) > 0:
        def get_subtype(row):
            dis = ""
            for disease in disease_cols:
                if row[disease] == 1:
                    dis = dis + "_" + disease.upper() if dis != "" else disease.upper()
            return 'AD_only' if dis == "" else dis
            
        df_val_meta['subtype'] = df_val_meta.apply(get_subtype, axis=1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=emb_val[:, 0], y=emb_val[:, 1], hue=df_val_meta[col_with_final_labels], palette='Set1', alpha=0.7)
        plt.title("Validation latent (coloured by replicated subtype)")
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=emb_val[:, 0], y=emb_val[:, 1], hue=df_val_meta['subtype'], palette='Set1', alpha=0.7)
        plt.title("Validation latent (coloured by comorbities)")
        plt.tight_layout()
        plt.savefig(f'{image_folder}/val_latent_by_subtype_and_disease.png')
        plt.close()

    # PLOT 3: Age / Sex
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=emb_val[:, 0], y=emb_val[:, 1], hue=df_val_meta[col_with_final_labels], palette='turbo', alpha=0.7)
    plt.title("Val latent (subtype)")
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=emb_val[:, 0], y=emb_val[:, 1], hue=df_val_meta['age_at_visit'], palette='turbo', alpha=0.7)
    plt.title("Val latent (age)")
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=emb_val[:, 0], y=emb_val[:, 1], hue=df_val_meta['sex'], alpha=0.7)
    plt.title("Val latent (sex)")
    plt.tight_layout()
    plt.savefig(f'{image_folder}/val_latent_by_subtype_and_age_and_sex.png')
    plt.close()

    # PLOT 4: Boxplot MMSE
    if cognitive_col in df_val_meta.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(x=col_with_final_labels, y=cognitive_col, data=df_val_meta)
        plt.title("Validation MMSE by replicated subtype")
        plt.savefig(f'{image_folder}/val_MMSE_by_replicated_subtype.png')
        plt.close()
