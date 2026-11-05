"""
Exploratory Data Analysis (EDA) for QSAR Chemical Biodegradation Dataset.
Inspects molecular descriptor distributions, missingness, class balance,
scale disparities, and linear correlations with biodegradability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "qsar_biodeg.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"
TARGET_COL = "biodeg_class"


def load_molecular_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the QSAR biodegradation dataset from disk."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    return pd.read_csv(csv_path)


def print_dataset_profile(mol_frame: pd.DataFrame) -> pd.Series:
    """Print dataset shape, data types summary, missing values, and target balance."""
    descriptor_cols = [c for c in mol_frame.columns if c != TARGET_COL]
    n_molecules, n_total_cols = mol_frame.shape

    print("=" * 72)
    print("QSAR BIODEGRADATION -- EXPLORATORY DATA PROFILE")
    print("=" * 72)
    print(f"Dataset Shape          : {n_molecules} molecules x {n_total_cols} columns")
    print(f"Input Descriptors      : {len(descriptor_cols)} features")
    print(f"Target Column          : '{TARGET_COL}'")

    dtype_breakdown = mol_frame[descriptor_cols].dtypes.value_counts()
    print("\n--- Descriptor Data Types ---")
    for dt_name, count in dtype_breakdown.items():
        print(f"  {str(dt_name):10s} : {count} descriptors")

    total_missing = int(mol_frame.isna().sum().sum())
    print(f"\n--- Missing Value Audit ---")
    print(f"  Total missing cells  : {total_missing}")

    class_counts = mol_frame[TARGET_COL].value_counts()
    class_pcts = mol_frame[TARGET_COL].value_counts(normalize=True) * 100.0
    print("\n--- Target Class Distribution ---")
    for label in class_counts.index:
        meaning = "Not Readily Biodegradable" if label == "NRB" else "Readily Biodegradable"
        print(f"  {label:4s} ({meaning:25s}): {class_counts[label]:4d} ({class_pcts[label]:5.2f}%)")

    # Binary target encoding (RB=1, NRB=0) to inspect point-biserial correlation
    binary_target = (mol_frame[TARGET_COL] == "RB").astype(int)
    corr_series = mol_frame[descriptor_cols].apply(lambda col: col.corr(binary_target))
    corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

    print("\n--- Top 10 Descriptors Correlated with Readily Biodegradable (RB=1) ---")
    for feat_name, r_val in corr_sorted.head(10).items():
        direction = "+ promotes RB" if r_val > 0 else "- inhibits RB"
        print(f"  {feat_name:12s} : r = {r_val:+.4f}  ({direction})")

    # Check high inter-feature collinearity pairs (|r| > 0.80)
    corr_matrix = mol_frame[descriptor_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_pairs = [
        (r_idx, c_idx, upper_tri.loc[r_idx, c_idx])
        for r_idx in upper_tri.index
        for c_idx in upper_tri.columns
        if upper_tri.loc[r_idx, c_idx] > 0.80
    ]
    collinear_pairs.sort(key=lambda item: item[2], reverse=True)
    print(f"\n--- High Inter-Descriptor Collinearity Pairs (|r| > 0.80): {len(collinear_pairs)} found ---")
    for f1, f2, r_abs in collinear_pairs[:8]:
        print(f"  {f1:12s} <-> {f2:12s} : |r| = {r_abs:.4f}")

    return corr_sorted


def export_class_balance_chart(mol_frame: pd.DataFrame, out_dir: Path) -> Path:
    """Save bar plot illustrating NRB vs RB class distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    counts = mol_frame[TARGET_COL].value_counts()
    palette = {"NRB": "#d95f02", "RB": "#1b9e77"}

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[palette.get(k, "#4c72b0") for k in counts.index],
        width=0.52,
        edgecolor="#1f2937",
        linewidth=1.1,
    )

    total = len(mol_frame)
    for bar in bars:
        height = int(bar.get_height())
        pct = (height / total) * 100.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 15,
            f"{height} molecules\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("QSAR Biodegradation Target Class Balance (n = 1,055)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Biodegradability Class", fontsize=10)
    ax.set_ylabel("Number of Chemical Compounds", fontsize=10)
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine()

    out_path = out_dir / "class_balance.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def export_correlation_heatmap(mol_frame: pd.DataFrame, corr_sorted: pd.Series, out_dir: Path) -> Path:
    """Save correlation heatmap of the top 12 biodegradability-linked molecular descriptors."""
    top_12_feats = corr_sorted.head(12).index.tolist()
    sub_df = mol_frame[top_12_feats].copy()
    sub_df["RB_Target"] = (mol_frame[TARGET_COL] == "RB").astype(int)

    corr_mat = sub_df.corr()
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=140)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

    sns.heatmap(
        corr_mat,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"shrink": 0.82, "label": "Pearson Correlation (r)"},
        linewidths=0.5,
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(
        "Inter-Correlation Among Top 12 Biodegradability Descriptors",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    out_path = out_dir / "top_descriptor_correlations.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def export_scale_disparity_plot(mol_frame: pd.DataFrame, out_dir: Path) -> Path:
    """Save chart of feature value spans showing why standardization is needed for KNN/SVM."""
    descriptor_cols = [c for c in mol_frame.columns if c != TARGET_COL]
    spans = (mol_frame[descriptor_cols].max() - mol_frame[descriptor_cols].min()).sort_values(ascending=True)

    # Pick 15 representative descriptors spanning smallest to largest dynamic ranges
    selected_idx = np.linspace(0, len(spans) - 1, 15, dtype=int)
    sampled_spans = spans.iloc[selected_idx]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)
    ax.barh(sampled_spans.index, sampled_spans.values, color="#3b82f6", edgecolor="#1e3a8a", height=0.68)
    ax.set_xscale("log")
    ax.set_title(
        "Dynamic Range Disparity Across Molecular Descriptors (Log Scale)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Feature Range (Max - Min, Log10 Scale)", fontsize=10)
    ax.set_ylabel("Molecular Descriptor", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    sns.despine()

    out_path = out_dir / "descriptor_scale_disparity.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    mol_frame = load_molecular_dataset(RAW_DATA_PATH)
    corr_sorted = print_dataset_profile(mol_frame)

    p1 = export_class_balance_chart(mol_frame, ASSETS_DIR)
    p2 = export_correlation_heatmap(mol_frame, corr_sorted, ASSETS_DIR)
    p3 = export_scale_disparity_plot(mol_frame, ASSETS_DIR)

    print("\n--- Saved EDA Plots ---")
    for path in (p1, p2, p3):
        size_kb = path.stat().st_size / 1024.0
        print(f"  [OK] {path.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")
    print("=" * 72)


if __name__ == "__main__":
    main()
