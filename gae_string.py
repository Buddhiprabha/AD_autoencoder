'''
This script trains a graph-smoothed autoencoder with a STRING prior.

SET the following variables before running.

- PARQUET_FILE: parquet file containing proteomic data for the discovery cohort AD samples
- METADATA_FILE: metadata file in CSV format containing patient metadata for the discovery cohort
- BATCH_COL_NAME: column containing batch effect data in the discovery cohort
- COGNITIVE_COL: column with cognitive test scores (e.g. MMSE)
- AGE_COL: column with age data
- ID_COL: column with patient ID data
'''

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers, losses, regularizers
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import random
import itertools

# ==========================================
# 0. REPRODUCIBILITY SETUP
# ==========================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)
print("Global random seeds set to 42.")

# define local random seed
RANDOM_STATE = 42

# ==========================================
# 1. CONFIGURATION & GRID SEARCH SETUP
# ==========================================
PARQUET_FILE = 'discovery_AD.parquet'
METADATA_FILE = 'metadata_discovery_AD.csv'

# define dataset columns used in the analysis
BATCH_COL_NAME = 'dis_batch_col' 
COGNITIVE_COL = 'cognitive_test_score'
AGE_COL = 'age_at_visit'
ID_COL = 'sample_id'

EPOCHS = 500

# define the values for hyperparameter grid
LATENT_DIMS = [32, 64, 128]          # Test 32, 64, and 128 node bottlenecks
L2_REGS = [1e-4, 1e-3]               # Standard vs Heavy weight penalty
DROPOUTS = [0.2, 0.3]                # Light vs Moderate dropout
LEARNING_RATES = [1e-3, 5e-4, 1e-4]
BATCH_SIZES = [16, 32, 64]
FUNNEL_DIMS = [256, 512]

# create the grid with all possible combinations
hyperparameter_grid = list(itertools.product(LATENT_DIMS, L2_REGS, DROPOUTS, LEARNING_RATES, BATCH_SIZES, FUNNEL_DIMS))
print(f"Total experimental setups to run: {len(hyperparameter_grid)}")

# ==========================================
# 2. CUSTOM LAYERS
# ==========================================
class GAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(GAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        # Unpack the smoothed input and the raw target
        x_smoothed, x_raw = data 
        with tf.GradientTape() as tape:
            # Encode using the graph-smoothed data
            z = self.encoder(x_smoothed, training=True)
            reconstruction = self.decoder(z, training=True)
            # Calculate loss against the RAW data
            recon_loss = tf.reduce_mean(tf.square(x_raw - reconstruction))
            reg_loss = sum(self.encoder.losses) 
            total_loss = recon_loss + reg_loss
            
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return {"mse": recon_loss}

    def test_step(self, data):
        x_smoothed, x_raw = data
        z = self.encoder(x_smoothed, training=False)
        reconstruction = self.decoder(z, training=False)
        # Calculate validation loss against the RAW data
        recon_loss = tf.reduce_mean(tf.square(x_raw - reconstruction))
        return {"mse": recon_loss}

class PrintEvery10Epochs(tf.keras.callbacks.Callback):
    def __init__(self):
        super(PrintEvery10Epochs, self).__init__()
        self.best_val = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_mse')
        
        # Track the best validation score
        if current_val < self.best_val:
            self.best_val = current_val
            
        # Print update every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Val MSE: {current_val:.4f} (Best: {self.best_val:.4f})")

# ==========================================
# STRING GRAPH CONSTRUCTION FUNCTION
# ==========================================
def build_string_adjacency(aptamer_list, analyte_file, string_links_file, string_aliases_file, min_score=700):
    print("--- Building Biological Prior Adjacency Matrix from STRING ---")
    
    # 1. Map Aptamers to Gene Symbols
    # load only the columns we need
    analyte_df = pd.read_csv(analyte_file, usecols=['apt_name', 'entrez_gene_symbol']).dropna()
    analyte_df['apt_name'] = analyte_df['apt_name'].str.replace('.', '_', regex=False)
    aptamer_to_gene = dict(zip(analyte_df['apt_name'], analyte_df['entrez_gene_symbol']))
    
    # Create a mapping of the current matrix indices to Gene Symbols
    idx_to_gene = {}
    for idx, apt_name in enumerate(aptamer_list):
        idx_to_gene[idx] = aptamer_to_gene.get(apt_name, None)

    # 2. Load STRING Aliases and create Gene -> STRING_ID mapping
    print("Loading STRING Aliases...")
    aliases_df = pd.read_csv(string_aliases_file, sep='\t')
    
    our_genes = set([g for g in idx_to_gene.values() if g is not None])
    filtered_aliases = aliases_df[aliases_df['alias'].isin(our_genes)]
    gene_to_string_id = dict(zip(filtered_aliases['alias'], filtered_aliases['#string_protein_id']))
    
    idx_to_string = {}
    string_to_idx = {}
    for idx, gene in idx_to_gene.items():
        if gene in gene_to_string_id:
            s_id = gene_to_string_id[gene]
            idx_to_string[idx] = s_id
            string_to_idx[s_id] = idx

    print(f"Successfully mapped {len(string_to_idx)} out of {len(aptamer_list)} proteins to STRING IDs.")

    # 3. Parse STRING Links and Build the Matrix
    print(f"Parsing STRING interactions (Min Score: {min_score})...")
    num_nodes = len(aptamer_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    chunksize = 1000000 
    edges_added = 0
    
    for chunk in pd.read_csv(string_links_file, sep=' ', chunksize=chunksize):
        chunk = chunk[chunk['combined_score'] >= min_score]
        valid_edges = chunk[chunk['protein1'].isin(string_to_idx) & chunk['protein2'].isin(string_to_idx)]
        
        # Vectorized NumPy indexing
        if not valid_edges.empty:
            # 1. Map the protein strings directly to integer indices
            idx1 = valid_edges['protein1'].map(string_to_idx).values
            idx2 = valid_edges['protein2'].map(string_to_idx).values
            
            # 2. Extract and normalize the weights
            weights = valid_edges['combined_score'].values / 1000.0
            
            # 3. Update the matrix in bulk
            adj_matrix[idx1, idx2] = weights
            adj_matrix[idx2, idx1] = weights
            
            edges_added += len(valid_edges)
            
    print(f"Graph construction complete! Added {edges_added} biologically validated edges.")
    
    # 4. Standard Graph Normalisation
    np.fill_diagonal(adj_matrix, 1.0)
    rowsum = np.array(adj_matrix.sum(1))
    # handle proteins with many connections (1/sqrt(degree))
    r_inv = np.power(rowsum, -0.5).flatten()
    # for proteins with no connections, avoid resulting in infinity
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    adj_normalized = r_mat_inv.dot(adj_matrix).dot(r_mat_inv)
    
    # Diagnostic: Check for orphaned proteins (degree == 1 because of self-loops)
    orphan_count = (adj_matrix.sum(axis=1) == 1.0).sum()
    print(f"At Min Score {min_score}:")
    print(f" -> Total Edges: {edges_added}")
    print(f" -> Orphaned Proteins (No neighbors): {orphan_count} out of {num_nodes}")
    
    return adj_normalized.astype(np.float32)
            
# ==========================================
# 3. DATA LOADING & GRAPH CONSTRUCTION
# ==========================================
def load_data():
    print(f"--- Loading Data ---")
    df_prot = pd.read_parquet(PARQUET_FILE)
    df_meta = pd.read_csv(METADATA_FILE).set_index("sample_id")
    
    # Infer batch via UMAP/KMeans
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    embedding = reducer.fit_transform(df_prot)
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    df_meta['inferred_draw_year'] = kmeans.fit_predict(embedding)
    
    # Filter MMSE & Age
    df_meta[COGNITIVE_COL] = pd.to_numeric(df_meta[COGNITIVE_COL], errors='coerce')
    df_meta[AGE_COL] = pd.to_numeric(df_meta[AGE_COL], errors='coerce')
    mask = (df_meta[COGNITIVE_COL] >= 15) & (df_meta[COGNITIVE_COL] <= 30) & (df_meta[AGE_COL] >= 65.0)
    df_meta = df_meta[mask]
    
    # Alignment
    common_ids = df_prot.index.intersection(df_meta.index)
    X = df_prot.loc[common_ids].values
    
    # Encode Batch
    raw_batch = df_meta.loc[common_ids, BATCH_COL_NAME].values
    y_batch = LabelEncoder().fit_transform(raw_batch)
    
    # Per-Batch Standardisation (Leakage check inside load_data builds graph only)
    # not doing the scaling here to avoid data leakage
    X_scaled = np.zeros_like(X)
    for b_id in np.unique(y_batch):
        indices = np.where(y_batch == b_id)[0]
        X_scaled[indices] = StandardScaler().fit_transform(X[indices])
        
    # Construct Normalised Adjacency Matrix
    print("Constructing Biological Prior Graph...")
    protein_names = df_prot.columns.tolist()
    
    adj_normalized = build_string_adjacency(
        aptamer_list=protein_names, 
        analyte_file='~/files/GNPC_HDS_general/HDS V1.3/AnalyteInfoV1_3.csv', 
        string_links_file='9606.protein.links.v12.0.txt', 
        string_aliases_file='9606.protein.aliases.v12.0.txt',
        min_score=700
    )
    
    # Return raw X for safe Train/Test split later
    return X, y_batch, adj_normalized.astype(np.float32), common_ids

# ==========================================
# 4. MODEL DEFINITIONS
# ==========================================
def build_encoder(num_nodes, latent_dim, funnel_dim, l2_reg, drop_rate):
    x_in  = Input(shape=(num_nodes, 1), name="Input_Features")
    hflat = layers.Flatten()(x_in)
    h     = layers.Dense(funnel_dim, activation='relu')(hflat)
    h     = layers.Dropout(drop_rate)(h)
    z     = layers.Dense(
        latent_dim,
        name="Latent_Z",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(h)
    return Model(x_in, z, name="Encoder")

def build_decoder(num_nodes, latent_dim, funnel_dim, drop_rate):
    z_in = Input(shape=(latent_dim,))
    
    # Use funnel_dim here!
    d1 = layers.Dense(funnel_dim, activation='relu')(z_in)
    d1 = layers.Dropout(drop_rate)(d1)
    
    d2 = layers.Dense(num_nodes, activation='linear')(d1)
    x_out = layers.Reshape((num_nodes, 1))(d2)
    return Model(z_in, x_out, name="Decoder")

# ==========================================
# 5. INITIAL SETUP (RUNS ONCE)
# ==========================================
X, y_batch, adj_matrix, sample_ids = load_data()
y_batch = y_batch.astype(np.float32)
num_proteins = X.shape[1]

# Train-Test Split & Leakage-Free Scaling
print("Splitting data into 80% Train and 20% Test...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_batch, test_size=0.2, random_state=RANDOM_STATE, stratify=y_batch
)

X_train = np.zeros_like(X_train_raw)
X_test = np.zeros_like(X_test_raw)
scalers = {} 

# per batch standardisation to avoid data leakage
for b_id in np.unique(y_train):
    train_mask = (y_train == b_id)
    test_mask = (y_test == b_id)
    scaler = StandardScaler()
    scaler.fit(X_train_raw[train_mask])
    scalers[b_id] = scaler
    X_train[train_mask] = scaler.transform(X_train_raw[train_mask])
    X_test[test_mask] = scaler.transform(X_test_raw[test_mask])

# Prepare full dataset for feature extraction at the end
X_full_scaled = np.zeros_like(X)
for b_id in np.unique(y_batch):
    batch_mask = (y_batch == b_id)
    if b_id in scalers:
        X_full_scaled[batch_mask] = scalers[b_id].transform(X[batch_mask])

# Because adj_matrix is symmetric, X @ A is mathematically equivalent 
# to doing A @ X^T for every individual patient.
print("Pre-computing Graph Convolutions...")
X_train_smoothed = X_train @ adj_matrix
X_test_smoothed = X_test @ adj_matrix
X_full_smoothed = X_full_scaled @ adj_matrix

# Reshape the SMOOTHED datasets (for the Encoder)
X_train_smoothed_resh = np.expand_dims(X_train_smoothed, axis=-1).astype(np.float32)
X_test_smoothed_resh = np.expand_dims(X_test_smoothed, axis=-1).astype(np.float32)
X_full_smoothed_resh = np.expand_dims(X_full_smoothed, axis=-1).astype(np.float32)

# Reshape the RAW datasets (for the Loss Function)
X_train_raw_resh = np.expand_dims(X_train, axis=-1).astype(np.float32)
X_test_raw_resh = np.expand_dims(X_test, axis=-1).astype(np.float32)

# ==========================================
# 6. AUTOMATED GRID SEARCH LOOP
# ==========================================
for current_latent, current_l2, current_drop, current_lr, current_batch, current_funnel in hyperparameter_grid:
    
    set_seeds(42)
    
    OUTPUT_DIR = f"Exp_Lat{current_latent}_L2{current_l2}_Drp{current_drop}_LR{current_lr}_Bch{current_batch}_Fnl{current_funnel}_STRING"
    
    # Check if the FINAL output file of this specific run already exists
    final_output_file = os.path.join(OUTPUT_DIR, "latent_proteomics_subtypes.csv")
    if os.path.exists(final_output_file):
        print(f"Skipping {OUTPUT_DIR} -> Already completed!")
        continue # Skips to the next hyperparameter combination
    
    print(f"\n{'='*60}")
    print(f"EXP: Lat={current_latent}|L2={current_l2}|Drop={current_drop}|LR={current_lr}|Batch={current_batch}|Funnel={current_funnel}")
    print(f"{'='*60}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create the datasets dynamically INSIDE the loop using current_batch
    # 1. Train Dataset (Needs Shuffling)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_smoothed_resh, X_train_raw_resh)) \
        .cache() \
        .shuffle(buffer_size=1024, seed=RANDOM_STATE) \
        .batch(current_batch) \
        .prefetch(tf.data.AUTOTUNE)
    
    # 2. Test Dataset (No Shuffling Needed)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_smoothed_resh, X_test_raw_resh)) \
        .cache() \
        .batch(current_batch) \
        .prefetch(tf.data.AUTOTUNE)
    
    np.save(os.path.join(OUTPUT_DIR, "training_adjacency_matrix.npy"), adj_matrix)

    # Build Models (pass current_funnel!)
    encoder = build_encoder(num_proteins, current_latent, current_funnel, current_l2, current_drop)
    decoder = build_decoder(num_proteins, current_latent, current_funnel, current_drop)
    
    gae_model = GAE(encoder, decoder)
    gae_model.compile(optimizer=optimizers.Adam(learning_rate=current_lr, clipnorm=1.0))

    weights_path = os.path.join(OUTPUT_DIR, "best_gae_encoder.weights.h5")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse', 
        patience=60, 
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    print_every_10 = PrintEvery10Epochs()

    # Train using the optimized C++ backend!
    history = gae_model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, print_every_10],
        verbose=0
    )

    encoder.save_weights(weights_path)

    # Save Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE', linestyle='--')
    plt.title(f'GAE History (Lat={current_latent}, L2={current_l2}, Drop={current_drop})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curve.png"))
    plt.close()

    # Extract & Save Latent Features
    print("Extracting features using best weights...")
    clean_features = encoder.predict(X_full_smoothed_resh, verbose=0)
    
    df_out = pd.DataFrame(clean_features, index=sample_ids)
    out_csv = os.path.join(OUTPUT_DIR, "latent_proteomics_subtypes.csv")
    df_out.to_csv(out_csv)
    print(f"Saved: {out_csv}")
    
    # clear TensorFlow Memory before the next grid search iteration!
    tf.keras.backend.clear_session()

print("\nGrid Search Complete! All models and features have been saved.")
