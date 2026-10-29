"""
This script checks for overlap of proteins between the CSF subtypes identified in Tjims et al. with
the proteomic bipolar axis.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ================================================================== CONFIG
TIJMS_FILE = "43587_2023_550_MOESM3_ESM.xlsx"
TIJMS_SHEET = "S table 4"
HEADER_ROW = 3                     # the row that yielded "Gene symbol" etc.
GENE_COL = "Gene symbol"
N_SUBTYPES = 5
SUBTYPE_NAMES = {1: "S1_neuronal_hyperplasticity", 2: "S2_innate_immune",
                 3: "S3_RNA_dysregulation", 4: "S4_choroid_plexus",
                 5: "S5_blood_brain_barrier"}
TIJMS_Q = 0.05

AD_DE = {c: f"pathways_discovery/DE_stats_Cluster_{c}_vs_Control.csv" for c in (0, 1, 2)}
AD_MAP = {"ELS": 0, "ELA": 1, "CCMS": 2}
FDR_T, FC_T = 0.05, 0.1
OUT = "tijms_overlap"


# ------------------------------------------------------------ plasma sets
def direction_sets(df):
    s = df[(df.FDR < FDR_T) & (df.Log2FC.abs() > FC_T)]
    g = s.groupby("Gene").Log2FC.apply(list)
    return ({k for k, v in g.items() if all(x > 0 for x in v)},
            {k for k, v in g.items() if all(x < 0 for x in v)},
            {k for k, v in g.items() if any(x > 0 for x in v)},
            {k for k, v in g.items() if any(x < 0 for x in v)})


def plasma_exclusive():
    de = {c: pd.read_csv(f) for c, f in AD_DE.items()}
    D = {c: direction_sets(d) for c, d in de.items()}
    out = {}
    for role, c in AD_MAP.items():
        up, dn = set(D[c][0]), set(D[c][1])
        for o in [x for x in de if x != c]:
            up -= D[o][2]; dn -= D[o][3]
        out[f"{role}_UP"] = {g.upper() for g in up}
        out[f"{role}_DOWN"] = {g.upper() for g in dn}
    measured = {str(g).upper() for g in de[0].Gene.dropna()}
    return out, measured


# ------------------------------------------------------------ column roles
def classify(block, name):
    """Return 'effect', 'pvalue' or 'unknown' for a block of columns."""
    v = pd.concat([pd.to_numeric(block[c], errors="coerce") for c in block],
                  ignore_index=True).dropna()
    if v.empty:
        return "unknown", {}
    stats_ = dict(min=v.min(), max=v.max(), frac_neg=(v < 0).mean(),
                  median=v.median())
    if v.min() >= -1e-9 and v.max() <= 1 + 1e-9:
        return "pvalue", stats_
    if stats_["frac_neg"] > 0.05:
        return "effect", stats_
    return "unknown", stats_


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_excel(TIJMS_FILE, sheet_name=TIJMS_SHEET, header=HEADER_ROW)
    print(f"loaded {t.shape[0]} proteins x {t.shape[1]} columns")
    if GENE_COL not in t.columns:
        sys.exit(f"'{GENE_COL}' not found. Columns start: {list(t.columns[:6])}")

    # the three blocks, by exact name pattern
    blocks = {}
    for suffix, label in [("", "block1"), (".1", "block2"), (".2", "block3")]:
        cols = [f"Controls vs subtype {s}{suffix}" for s in range(1, N_SUBTYPES + 1)]
        missing = [c for c in cols if c not in t.columns]
        if missing:
            sys.exit(f"missing columns for {label}: {missing}")
        blocks[label] = cols

    print("\nidentifying what each block contains:")
    roles = {}
    for label, cols in blocks.items():
        role, st = classify(t[cols], label)
        roles[label] = role
        print(f"  {label}: {role:<8} range [{st.get('min', np.nan):.3g}, "
              f"{st.get('max', np.nan):.3g}]  {100*st.get('frac_neg', 0):.0f}% negative")

    eff = [l for l, r in roles.items() if r == "effect"]
    pv = [l for l, r in roles.items() if r == "pvalue"]
    if not eff or not pv:
        sys.exit("could not identify an effect-size block and a p-value block; "
                 "inspect the columns manually")
    eff_block = eff[0]

    # if two p-like blocks, the adjusted one is elementwise >= the raw one
    if len(pv) >= 2:
        a = pd.to_numeric(t[blocks[pv[0]][0]], errors="coerce")
        b = pd.to_numeric(t[blocks[pv[1]][0]], errors="coerce")
        ok = a.notna() & b.notna()
        p_block = pv[0] if (a[ok] <= b[ok]).mean() > 0.5 else pv[1]
        q_block = pv[1] if p_block == pv[0] else pv[0]
        print(f"  -> effect = {eff_block}, raw p = {p_block}, adjusted p = {q_block}")
    else:
        q_block = pv[0]
        print(f"  -> effect = {eff_block}, p = {q_block} (only one p-like block)")

    t[GENE_COL] = t[GENE_COL].astype(str).str.upper().str.strip()
    t = t[t[GENE_COL].notna() & (t[GENE_COL] != "NAN")]
    csf_measured = set(t[GENE_COL])

    csf = {}
    print("\nCSF subtype sets (adjusted p < %.2f):" % TIJMS_Q)
    for s in range(1, N_SUBTYPES + 1):
        fc = pd.to_numeric(t[blocks[eff_block][s - 1]], errors="coerce")
        q = pd.to_numeric(t[blocks[q_block][s - 1]], errors="coerce")
        nm = SUBTYPE_NAMES.get(s, f"S{s}")
        csf[f"{nm}_UP"] = set(t.loc[(q < TIJMS_Q) & (fc < 0), GENE_COL])
        csf[f"{nm}_DOWN"] = set(t.loc[(q < TIJMS_Q) & (fc > 0), GENE_COL])
        print(f"  {nm:<30}{len(csf[nm+'_UP']):>5} up  {len(csf[nm+'_DOWN']):>5} down")

    plasma, plasma_measured = plasma_exclusive()
    background = csf_measured & plasma_measured
    print(f"\nbackground measurable in BOTH studies: {len(background)}")
    print(f"  (CSF {len(csf_measured)}, plasma {len(plasma_measured)})")
    if len(background) < 200:
        print("  WARNING: small shared background; the statistics will be noisy")

    rows = []
    for pk in [k for k in plasma if k.startswith(("ELA", "ELS"))]:
        A = plasma[pk] & background
        for ck, cv in csf.items():
            B = cv & background
            if len(A) < 5 or len(B) < 5:
                continue
            inter = A & B
            a = len(inter); b, c = len(B) - a, len(A) - a
            d = len(background) - a - b - c
            orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            same = ("UP" in pk) == ("UP" in ck)
            rows.append(dict(plasma=pk, csf=ck,
                             direction="same" if same else "opposite",
                             n_plasma=len(A), n_csf=len(B), overlap=a,
                             overlap_coef=round(a / min(len(A), len(B)), 3),
                             jaccard=round(a / len(A | B), 3),
                             odds_ratio=round(orr, 2), p=p,
                             shared_genes=";".join(sorted(inter)[:50])))
    r = pd.DataFrame(rows)
    if r.empty:
        sys.exit("no comparable sets")
    r["q"] = multipletests(r.p, method="fdr_bh")[1]
    r = r.sort_values("p")
    r.to_csv(f"{OUT}/tijms_overlap.csv", index=False)

    print("\n" + "=" * 92)
    print("OVERLAP OF PLASMA ELA / ELS WITH CSF SUBTYPES")
    print("=" * 92)
    print(f"  {'plasma':<11}{'CSF subtype':<34}{'dir':<10}{'n_p':>5}{'n_c':>6}"
          f"{'ovl':>6}{'coef':>7}{'OR':>7}{'q':>10}")
    print("  " + "-" * 88)
    for _, x in r.head(20).iterrows():
        flag = " *" if x.q < 0.05 else ""
        print(f"  {x.plasma:<11}{x.csf[:32]:<34}{x.direction:<10}{x.n_plasma:>5}"
              f"{x.n_csf:>6}{x.overlap:>6}{x.overlap_coef:>7.2f}"
              f"{x.odds_ratio:>7.2f}{x.q:>10.2g}{flag}")
    print(f"\nwritten: {OUT}/tijms_overlap.csv")

if __name__ == "__main__":
    main()
