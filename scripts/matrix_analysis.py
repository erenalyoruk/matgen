#!/usr/bin/env python3
"""
matrix_analysis.py - Comprehensive Sparse Matrix Feature Analysis

Computes all structural features from Table 3 and generates comparison reports.
Can analyze MatrixMarket files or scipy.sparse matrices.
Includes visualization capabilities for sparsity pattern comparison.
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import argparse

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LogNorm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Visualization features disabled.")
    print("Install with: pip install matplotlib")


@dataclass
class MatrixFeatures:
    """Structural features of a sparse matrix (matching Table 3)"""
    N: int              # Matrix dimension
    NNZ: int            # Number of nonzeros
    Dens: float         # Density (NNZ / N^2)
    Psym: float         # Pattern symmetry (ratio of symmetric pairs)
    Diag: int           # Nonzeros on main diagonal
    Ndg: int            # Number of diagonals with at least one nonzero
    Dist: float         # Average distance from diagonal (normalized)
    Band: int           # Bandwidth (max diagonal distance)
    Profil: int         # Profile (sum of distances)
    Rmx: int            # Row-wise max nonzeros
    Rmi: int            # Row-wise min nonzeros (excluding empty)
    Rstd: float         # Row-wise standard deviation
    Cmx: int            # Column-wise max nonzeros
    Cmi: int            # Column-wise min nonzeros (excluding empty)
    Cstd: float         # Column-wise standard deviation
    generation_time_ms: float = 0.0  # Generation time in milliseconds
    
    def __str__(self):
        """Print in table format"""
        return (f"{self.N:5d} {self.NNZ:6d} {self.Dens:5.2f} {self.Psym:5.2f} "
                f"{self.Diag:4d} {self.Ndg:4d} {self.Dist:5.1f} {self.Band:4d} "
                f"{self.Profil:6d} {self.Rmx:4d} {self.Rmi:4d} {self.Rstd:5.1f} "
                f"{self.Cmx:4d} {self.Cmi:4d} {self.Cstd:5.1f}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_sparsity_pattern(matrix: sp.spmatrix, ax=None, title: str = None, 
                          color: str = '#1f77b4', markersize: float = None,
                          show_stats: bool = True):
    """
    Plot the sparsity pattern of a sparse matrix (spy plot).
    
    Args:
        matrix: Sparse matrix to visualize
        ax: Matplotlib axes (creates new figure if None)
        title: Plot title
        color: Color for nonzero markers
        markersize: Size of markers (auto-calculated if None)
        show_stats: Whether to show NNZ count in subtitle
    
    Returns:
        Figure and axes objects
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required for visualization")
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()
    
    # Convert to COO for plotting
    coo = sp.coo_matrix(matrix)
    
    # Auto-calculate marker size based on matrix dimension
    if markersize is None:
        # Smaller markers for larger matrices
        markersize = max(0.1, min(2.0, 500.0 / coo.shape[0]))
    
    # Plot nonzeros
    ax.scatter(coo.col, coo.row, s=markersize, c=color, marker='s', 
               linewidths=0, alpha=0.8)
    
    # Set axis properties
    ax.set_xlim(-0.5, coo.shape[1] - 0.5)
    ax.set_ylim(coo.shape[0] - 0.5, -0.5)  # Invert y-axis
    ax.set_aspect('equal')
    
    # Labels
    ax.set_xlabel('Columns', fontsize=10)
    ax.set_ylabel('Rows', fontsize=10)
    
    if title:
        if show_stats:
            ax.set_title(f"{title}\nNumber of nonzeros: {coo.nnz:,}", fontsize=11)
        else:
            ax.set_title(title, fontsize=11)
    
    ax.tick_params(labelsize=8)
    
    return fig, ax


def plot_comparison(original: sp.spmatrix, generated: sp.spmatrix,
                    orig_name: str = "Original", gen_name: str = "Generated",
                    method_name: str = "Lanczos", figsize: Tuple[int, int] = (14, 6),
                    save_path: str = None, dpi: int = 150):
    """
    Create side-by-side comparison plot of original and generated matrices.
    Similar to Figure 2, 3, 4 in the MatGen paper.
    
    Args:
        original: Original sparse matrix
        generated: Generated sparse matrix
        orig_name: Label for original matrix
        gen_name: Label for generated matrix
        method_name: Name of the generation method
        figsize: Figure size (width, height)
        save_path: Path to save figure (displays if None)
        dpi: Resolution for saved figure
    
    Returns:
        Figure object
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required for visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original
    plot_sparsity_pattern(original, ax=axes[0], title=f"(A) {orig_name}",
                          color='#1f77b4')
    
    # Plot generated
    plot_sparsity_pattern(generated, ax=axes[1], title=f"(B) {gen_name} ({method_name})",
                          color='#1f77b4')
    
    # Add scale information
    scale = generated.shape[0] / original.shape[0]
    fig.suptitle(f"Sparsity Pattern Comparison (Scale: {scale:.2f}×)", 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved comparison plot to {save_path}")
    
    return fig


def plot_multi_comparison(original: sp.spmatrix, generated_list: List[sp.spmatrix],
                          method_names: List[str], orig_name: str = "Original",
                          ncols: int = 4, figsize_per_plot: Tuple[float, float] = (4, 4),
                          save_path: str = None, dpi: int = 150):
    """
    Create multi-panel comparison plot like Figure 2 in MatGen paper.
    Shows original and multiple generated versions.
    
    Args:
        original: Original sparse matrix
        generated_list: List of generated matrices
        method_names: List of method names for each generated matrix
        orig_name: Label for original matrix
        ncols: Number of columns in the grid
        figsize_per_plot: Size of each subplot
        save_path: Path to save figure
        dpi: Resolution
    
    Returns:
        Figure object
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required for visualization")
    
    n_plots = 1 + len(generated_list)  # Original + all generated
    nrows = (n_plots + ncols - 1) // ncols
    
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot original
    plot_sparsity_pattern(original, ax=axes_flat[0], title=f"(A) {orig_name}",
                          color='#1f77b4')
    
    # Plot each generated matrix
    labels = 'BCDEFGHIJKLMNOP'
    for i, (gen_matrix, method) in enumerate(zip(generated_list, method_names)):
        label = labels[i] if i < len(labels) else str(i+1)
        plot_sparsity_pattern(gen_matrix, ax=axes_flat[i+1], 
                              title=f"({label}) {method}",
                              color='#1f77b4')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved multi-comparison plot to {save_path}")
    
    return fig


def plot_feature_comparison_bars(orig_features: 'MatrixFeatures', 
                                  gen_features: 'MatrixFeatures',
                                  scale: float, nnz_scale_power: float = 1.2,
                                  save_path: str = None, dpi: int = 150):
    """
    Create bar chart comparing normalized feature values.
    Shows how well each feature matches the expected scaled value.
    
    Args:
        orig_features: Features of original matrix
        gen_features: Features of generated matrix
        scale: Scale factor (new_size / original_size)
        nnz_scale_power: Power used for NNZ scaling
        save_path: Path to save figure
        dpi: Resolution
    
    Returns:
        Figure object
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required for visualization")
    
    # Calculate expected values
    nnz_scale = scale ** nnz_scale_power
    linear_scale = scale
    quadratic_scale = scale ** 2
    nnz_per_row_scale = scale ** (nnz_scale_power - 1)
    
    # Feature names and their expected/actual ratios
    features = [
        ('NNZ', gen_features.NNZ / (orig_features.NNZ * nnz_scale)),
        ('Density', gen_features.Dens / (orig_features.Dens * nnz_scale / quadratic_scale)),
        ('Symmetry', gen_features.Psym / orig_features.Psym if orig_features.Psym > 0 else 1.0),
        ('Bandwidth', gen_features.Band / (orig_features.Band * linear_scale)),
        ('Diag NNZ', gen_features.Diag / (orig_features.Diag * linear_scale)),
        ('Num Diag', gen_features.Ndg / (orig_features.Ndg * linear_scale)),
        ('Row Max', gen_features.Rmx / (orig_features.Rmx * nnz_per_row_scale)),
        ('Row Std', gen_features.Rstd / (orig_features.Rstd * nnz_per_row_scale) if orig_features.Rstd > 0 else 1.0),
        ('Col Max', gen_features.Cmx / (orig_features.Cmx * nnz_per_row_scale)),
        ('Col Std', gen_features.Cstd / (orig_features.Cstd * nnz_per_row_scale) if orig_features.Cstd > 0 else 1.0),
    ]
    
    names = [f[0] for f in features]
    ratios = [f[1] for f in features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(names))
    width = 0.6
    
    # Color bars based on how close to 1.0 (perfect)
    colors = []
    for r in ratios:
        diff = abs(r - 1.0)
        if diff < 0.1:
            colors.append('#2ecc71')  # Green - excellent
        elif diff < 0.25:
            colors.append('#f39c12')  # Orange - OK
        else:
            colors.append('#e74c3c')  # Red - poor
    
    bars = ax.bar(x, ratios, width, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add reference line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Perfect (1.0)')
    
    # Add tolerance bands
    ax.axhspan(0.9, 1.1, alpha=0.2, color='green', label='±10% tolerance')
    ax.axhspan(0.75, 0.9, alpha=0.1, color='orange')
    ax.axhspan(1.1, 1.25, alpha=0.1, color='orange')
    
    # Labels
    ax.set_xlabel('Feature', fontsize=11)
    ax.set_ylabel('Generated / Expected Ratio', fontsize=11)
    ax.set_title(f'Feature Preservation Analysis (Scale: {scale:.2f}×, NNZ Power: {nnz_scale_power})',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(0, max(2.0, max(ratios) * 1.1))
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.annotate(f'{ratio:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='< 10% diff'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='10-25% diff'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='> 25% diff'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved feature comparison plot to {save_path}")
    
    return fig


def plot_density_heatmap(matrix: sp.spmatrix, ax=None, title: str = None,
                         bins: int = 50, cmap: str = 'Blues', 
                         log_scale: bool = True):
    """
    Create a density heatmap showing concentration of nonzeros.
    Useful for large matrices where individual points aren't visible.
    
    Args:
        matrix: Sparse matrix
        ax: Matplotlib axes
        title: Plot title
        bins: Number of bins in each dimension
        cmap: Colormap name
        log_scale: Use logarithmic color scale
    
    Returns:
        Figure and axes
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required for visualization")
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()
    
    coo = sp.coo_matrix(matrix)
    
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        coo.col, coo.row, 
        bins=bins, 
        range=[[0, coo.shape[1]], [0, coo.shape[0]]]
    )
    
    # Plot heatmap
    if log_scale:
        # Add small value to avoid log(0)
        heatmap = np.where(heatmap > 0, heatmap, np.nan)
        im = ax.imshow(heatmap.T, origin='upper', cmap=cmap,
                       extent=[0, coo.shape[1], coo.shape[0], 0],
                       aspect='equal', norm=LogNorm(vmin=1))
    else:
        im = ax.imshow(heatmap.T, origin='upper', cmap=cmap,
                       extent=[0, coo.shape[1], coo.shape[0], 0],
                       aspect='equal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Nonzeros per cell', fontsize=10)
    
    ax.set_xlabel('Columns', fontsize=10)
    ax.set_ylabel('Rows', fontsize=10)
    
    if title:
        ax.set_title(f"{title}\nNNZ: {coo.nnz:,}", fontsize=11)
    
    return fig, ax


def plot_comprehensive_analysis(original: sp.spmatrix, generated: sp.spmatrix,
                                 orig_features: 'MatrixFeatures',
                                 gen_features: 'MatrixFeatures',
                                 method_name: str = "Lanczos",
                                 nnz_scale_power: float = 1.2,
                                 save_path: str = None, dpi: int = 150):
    """
    Create comprehensive analysis figure with multiple panels:
    - Sparsity patterns (original vs generated)
    - Density heatmaps
    - Feature comparison bars
    
    Args:
        original: Original matrix
        generated: Generated matrix
        orig_features: Computed features of original
        gen_features: Computed features of generated
        method_name: Generation method name
        nnz_scale_power: NNZ scaling power
        save_path: Path to save figure
        dpi: Resolution
    
    Returns:
        Figure object
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required for visualization")
    
    scale = gen_features.N / orig_features.N
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.25)
    
    # Row 1: Sparsity patterns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    plot_sparsity_pattern(original, ax=ax1, title="(A) Original")
    plot_sparsity_pattern(generated, ax=ax2, title=f"(B) {method_name}")
    
    # Row 2: Density heatmaps (useful for large matrices)
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Determine appropriate bin count based on matrix size
    bins_orig = min(100, max(20, orig_features.N // 20))
    bins_gen = min(100, max(20, gen_features.N // 20))
    
    plot_density_heatmap(original, ax=ax3, title="(C) Original Density", bins=bins_orig)
    plot_density_heatmap(generated, ax=ax4, title=f"(D) {method_name} Density", bins=bins_gen)
    
    # Row 3: Feature comparison (spans both columns)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Compute ratios for bar chart
    nnz_scale = scale ** nnz_scale_power
    linear_scale = scale
    quadratic_scale = scale ** 2
    nnz_per_row_scale = scale ** (nnz_scale_power - 1)
    
    features = [
        ('NNZ', gen_features.NNZ / (orig_features.NNZ * nnz_scale) if orig_features.NNZ > 0 else 1),
        ('Density', gen_features.Dens / (orig_features.Dens * nnz_scale / quadratic_scale) if orig_features.Dens > 0 else 1),
        ('Symmetry', gen_features.Psym / orig_features.Psym if orig_features.Psym > 0 else 1),
        ('Bandwidth', gen_features.Band / (orig_features.Band * linear_scale) if orig_features.Band > 0 else 1),
        ('Diag NNZ', gen_features.Diag / (orig_features.Diag * linear_scale) if orig_features.Diag > 0 else 1),
        ('Num Diag', gen_features.Ndg / (orig_features.Ndg * linear_scale) if orig_features.Ndg > 0 else 1),
        ('Row Max', gen_features.Rmx / (orig_features.Rmx * nnz_per_row_scale) if orig_features.Rmx > 0 else 1),
        ('Col Max', gen_features.Cmx / (orig_features.Cmx * nnz_per_row_scale) if orig_features.Cmx > 0 else 1),
    ]
    
    names = [f[0] for f in features]
    ratios = [min(f[1], 2.5) for f in features]  # Cap at 2.5 for visualization
    
    x = np.arange(len(names))
    colors = ['#2ecc71' if abs(r-1) < 0.15 else '#f39c12' if abs(r-1) < 0.3 else '#e74c3c' 
              for r in ratios]
    
    bars = ax5.bar(x, ratios, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    ax5.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
    ax5.axhspan(0.85, 1.15, alpha=0.2, color='green')
    
    ax5.set_xlabel('Feature', fontsize=11)
    ax5.set_ylabel('Generated / Expected', fontsize=11)
    ax5.set_title('(E) Feature Preservation Ratios (1.0 = Perfect)', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(names, rotation=45, ha='right')
    ax5.set_ylim(0, max(2.0, max(ratios) * 1.1))
    
    for bar, ratio in zip(bars, ratios):
        ax5.annotate(f'{ratio:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Main title
    fig.suptitle(f'Sparse Matrix Scaling Analysis: {orig_features.N}→{gen_features.N} (Scale: {scale:.2f}×)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Saved comprehensive analysis to {save_path}")
    
    return fig


def compute_features(matrix: sp.spmatrix) -> MatrixFeatures:
    """
    Compute all structural features from a sparse matrix.
    
    Args:
        matrix: Sparse matrix in any scipy.sparse format
        
    Returns:
        MatrixFeatures object with all computed features
    """
    # Convert to COO for easier analysis
    coo = sp.coo_matrix(matrix)
    csr = sp.csr_matrix(matrix)
    csc = sp.csc_matrix(matrix)
    
    N = coo.shape[0]
    assert coo.shape[0] == coo.shape[1], "Matrix must be square"
    
    NNZ = coo.nnz
    
    # Density
    Dens = NNZ / (N * N)
    
    # Diagonal count
    Diag = 0
    diagonal_set = set()
    for r, c in zip(coo.row, coo.col):
        if r == c:
            Diag += 1
        diagonal_set.add(c - r)
    
    # Number of diagonals with at least one nonzero
    Ndg = len(diagonal_set)
    
    # Pattern symmetry: count symmetric pairs
    coords = set(zip(coo.row, coo.col))
    symmetric_pairs = 0
    for r, c in coords:
        if r < c and (c, r) in coords:  # Only count upper triangle
            symmetric_pairs += 1
    
    off_diagonal = NNZ - Diag
    Psym = (symmetric_pairs * 2 / off_diagonal) if off_diagonal > 0 else 1.0
    
    # Distance and bandwidth statistics
    distances = np.abs(coo.row - coo.col)
    Dist = np.mean(distances) / N if NNZ > 0 else 0.0
    Band = int(np.max(distances)) if NNZ > 0 else 0
    
    # Profile: sum of distances from start of each row to first nonzero
    Profil = 0
    for r in range(N):
        row_start = csr.indptr[r]
        row_end = csr.indptr[r + 1]
        if row_end > row_start:
            first_col = csr.indices[row_start]
            Profil += abs(r - first_col)
    
    # Row statistics
    row_nnz = np.diff(csr.indptr)
    nonempty_rows = row_nnz[row_nnz > 0]
    
    if len(nonempty_rows) > 0:
        Rmx = int(np.max(nonempty_rows))
        Rmi = int(np.min(nonempty_rows))
        Rstd = float(np.std(row_nnz))
    else:
        Rmx = Rmi = 0
        Rstd = 0.0
    
    # Column statistics
    col_nnz = np.diff(csc.indptr)
    nonempty_cols = col_nnz[col_nnz > 0]
    
    if len(nonempty_cols) > 0:
        Cmx = int(np.max(nonempty_cols))
        Cmi = int(np.min(nonempty_cols))
        Cstd = float(np.std(col_nnz))
    else:
        Cmx = Cmi = 0
        Cstd = 0.0
    
    return MatrixFeatures(
        N=N, NNZ=NNZ, Dens=Dens, Psym=Psym, Diag=Diag, Ndg=Ndg,
        Dist=Dist, Band=Band, Profil=Profil,
        Rmx=Rmx, Rmi=Rmi, Rstd=Rstd,
        Cmx=Cmx, Cmi=Cmi, Cstd=Cstd
    )


def print_header():
    """Print table header matching Table 3 format"""
    print(f"{'Method':<12} {'N':>5} {'NNZ':>6} {'Dens':>5} {'Psym':>5} "
          f"{'Diag':>4} {'Ndg':>4} {'Dist':>5} {'Band':>4} {'Profil':>6} "
          f"{'Rmx':>4} {'Rmi':>4} {'Rstd':>5} {'Cmx':>4} {'Cmi':>4} {'Cstd':>5} "
          f"{'Time':>10}")
    print("-" * 110)


def print_comparison(orig: MatrixFeatures, gen: MatrixFeatures, method_name: str = "Lanczos", 
                     nnz_scale_power: float = 1.2):
    """
    Print detailed comparison between original and generated matrices.
    
    IMPORTANT: Features are normalized by the scale factor before comparison.
    This is because when scaling a matrix, many features should scale proportionally.
    
    Feature scaling expectations:
    - NNZ: scales with scale^nnz_scale_power (configurable, default 1.2)
    - Density: depends on NNZ scaling (for power=2, stays same; for power<2, decreases)
    - Pattern Symmetry: dimensionless, should stay same
    - Diagonal NNZ: scales linearly with N
    - Num Diagonals: scales linearly with N (approximately)
    - Avg Distance (normalized): should stay same (already normalized by N)
    - Bandwidth: scales linearly with N
    - Profile: scales with ~N * bandwidth, roughly N^2
    - Row/Col Max: scales with scale^(nnz_power-1) = avg entries per row scaling
    - Row/Col Min: harder to predict, use same scaling with more tolerance
    - Row/Col Std Dev: scales similarly to average entries per row
    """
    
    scale = gen.N / orig.N
    
    def rel_diff(orig_val, expected_val, gen_val):
        """Calculate relative difference from expected value"""
        if expected_val == 0:
            return 0.0 if gen_val == 0 else 100.0
        return 100.0 * abs(gen_val - expected_val) / abs(expected_val)
    
    def rel_diff_simple(val1, val2):
        """Simple relative difference between two values"""
        if val1 == 0:
            return 0.0 if val2 == 0 else 100.0
        return 100.0 * abs(val2 - val1) / abs(val1)
    
    print("\n" + "="*85)
    print(f"FEATURE PRESERVATION ANALYSIS (Scale: {scale:.2f}x, NNZ power: {nnz_scale_power})")
    print("="*85)
    
    print(f"\n{'Feature':<20} | {'Original':>10} | {'Expected':>10} | {'Generated':>10} | {'Diff %':>8} | {'Status':>8}")
    print("-" * 85)
    
    def print_row(name, orig_val, expected_val, gen_val, threshold=10.0):
        diff = rel_diff(orig_val, expected_val, gen_val)
        if diff < threshold:
            status = "✓ GOOD"
        elif diff < 30.0:
            status = "~ OK"
        else:
            status = "✗ POOR"
        
        print(f"{name:<20} | {orig_val:>10.2f} | {expected_val:>10.2f} | {gen_val:>10.2f} | {diff:>7.1f}% | {status:>8}")
    
    # Calculate expected values based on scaling
    nnz_scale = scale ** nnz_scale_power
    linear_scale = scale
    quadratic_scale = scale ** 2
    
    # Row/col statistics scaling: 
    # - Average NNZ per row = total_NNZ / N = orig_nnz * scale^power / (orig_N * scale) 
    #   = orig_avg * scale^(power-1)
    # For power=1.2: scale^0.2 (very slight increase)
    # For power=1.0: scale^0 = 1 (no change)
    # For power=2.0: scale^1 = linear
    nnz_per_row_scale = scale ** (nnz_scale_power - 1.0)
    
    # Expected values
    exp_nnz = orig.NNZ * nnz_scale
    exp_dens = orig.Dens * (nnz_scale / quadratic_scale)  # NNZ/N^2
    exp_diag = orig.Diag * linear_scale
    exp_ndg = min(orig.Ndg * linear_scale, 2 * gen.N - 1)  # Can't exceed max diagonals
    exp_band = orig.Band * linear_scale
    exp_profil = orig.Profil * quadratic_scale  # Roughly N^2 scaling
    
    # Row/col stats scale with average entries per row
    exp_rmx = orig.Rmx * nnz_per_row_scale
    exp_rmi = max(1, orig.Rmi * nnz_per_row_scale)  # At least 1
    exp_rstd = orig.Rstd * nnz_per_row_scale
    exp_cmx = orig.Cmx * nnz_per_row_scale
    exp_cmi = max(1, orig.Cmi * nnz_per_row_scale)
    exp_cstd = orig.Cstd * nnz_per_row_scale
    
    # Print comparisons
    print_row("NNZ", orig.NNZ, exp_nnz, gen.NNZ, 15.0)
    print_row("Density", orig.Dens, exp_dens, gen.Dens, 20.0)
    print_row("Pattern Symmetry", orig.Psym, orig.Psym, gen.Psym, 10.0)  # Should stay same
    print_row("Diagonal NNZ", orig.Diag, exp_diag, gen.Diag, 25.0)
    print_row("Num Diagonals", orig.Ndg, exp_ndg, gen.Ndg, 25.0)
    print_row("Avg Distance", orig.Dist, orig.Dist, gen.Dist, 15.0)  # Already normalized
    print_row("Bandwidth", orig.Band, exp_band, gen.Band, 15.0)
    print_row("Profile", orig.Profil, exp_profil, gen.Profil, 30.0)
    print_row("Row Max", orig.Rmx, exp_rmx, gen.Rmx, 40.0)  # More tolerance
    print_row("Row Min", orig.Rmi, exp_rmi, gen.Rmi, 60.0)  # Much more tolerance for min
    print_row("Row Std Dev", orig.Rstd, exp_rstd, gen.Rstd, 40.0)
    print_row("Col Max", orig.Cmx, exp_cmx, gen.Cmx, 40.0)
    print_row("Col Min", orig.Cmi, exp_cmi, gen.Cmi, 60.0)
    print_row("Col Std Dev", orig.Cstd, exp_cstd, gen.Cstd, 40.0)
    
    # Overall quality score - use key structural features
    # Weight the most important structural features more heavily
    diffs = [
        rel_diff(orig.Psym, orig.Psym, gen.Psym),           # Symmetry (should stay same)
        rel_diff(orig.Dist, orig.Dist, gen.Dist),           # Normalized distance
        rel_diff(orig.Band, exp_band, gen.Band),            # Bandwidth
        rel_diff(orig.NNZ, exp_nnz, gen.NNZ),               # NNZ
        rel_diff(orig.Diag, exp_diag, gen.Diag),            # Diagonal NNZ
        rel_diff(orig.Rstd, exp_rstd, gen.Rstd) * 0.5,      # Row distribution (half weight)
    ]
    avg_diff = np.mean(diffs)
    
    print("-" * 85)
    print(f"{'OVERALL SCORE':<20} | {'':<10} | {'':<10} | {'':<10} | {avg_diff:>7.1f}% | ", end="")
    
    if avg_diff < 10.0:
        print("★ EXCELLENT")
    elif avg_diff < 20.0:
        print("✓ GOOD")
    elif avg_diff < 30.0:
        print("~ OK")
    else:
        print("✗ POOR")
    
    if gen.generation_time_ms > 0:
        print(f"\nGeneration time: {gen.generation_time_ms:.2f} ms")
        throughput = (gen.NNZ / gen.generation_time_ms) * 1000.0
        print(f"Throughput: {throughput:,.0f} NNZ/sec")
    
    print("="*85)


def load_matrix_market(filepath: str) -> sp.csr_matrix:
    """Load a MatrixMarket file"""
    return sp.csr_matrix(sio.mmread(filepath))


def analyze_files(original_path: str, generated_path: str, method_name: str = "Lanczos",
                  nnz_scale_power: float = 1.2, plot: bool = False, 
                  plot_path: str = None, show_plot: bool = True):
    """Analyze and compare two matrix files"""
    
    print(f"Loading matrices...")
    orig_matrix = load_matrix_market(original_path)
    gen_matrix = load_matrix_market(generated_path)
    
    print(f"Computing features...")
    orig_features = compute_features(orig_matrix)
    gen_features = compute_features(gen_matrix)
    
    # Print table
    print("\n" + "="*110)
    print_header()
    print(f"{'Original':<12} {orig_features}")
    print(f"{method_name:<12} {gen_features}")
    
    # Print comparison with scale-aware normalization
    print_comparison(orig_features, gen_features, method_name, nnz_scale_power)
    
    # Generate visualizations if requested
    if plot and HAS_MATPLOTLIB:
        print("\nGenerating visualizations...")
        
        # Determine output path
        if plot_path is None:
            orig_stem = Path(original_path).stem
            gen_stem = Path(generated_path).stem
            plot_path = f"comparison_{orig_stem}_vs_{gen_stem}.png"
        
        # Generate comprehensive analysis plot
        fig = plot_comprehensive_analysis(
            orig_matrix, gen_matrix,
            orig_features, gen_features,
            method_name=method_name,
            nnz_scale_power=nnz_scale_power,
            save_path=plot_path
        )
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
    elif plot and not HAS_MATPLOTLIB:
        print("\nWarning: Cannot generate plots - matplotlib not installed")
    
    return orig_features, gen_features, orig_matrix, gen_matrix


def save_comparison_csv(orig: MatrixFeatures, gen: MatrixFeatures, output_path: str):
    """Save comparison to CSV file"""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Original', 'Generated', 'Rel_Diff_%'])
        
        def write_row(name, orig_val, gen_val):
            diff = 100.0 * abs(gen_val - orig_val) / abs(orig_val) if orig_val != 0 else 0.0
            writer.writerow([name, orig_val, gen_val, f"{diff:.2f}"])
        
        write_row('N', orig.N, gen.N)
        write_row('NNZ', orig.NNZ, gen.NNZ)
        write_row('Dens', orig.Dens, gen.Dens)
        write_row('Psym', orig.Psym, gen.Psym)
        write_row('Diag', orig.Diag, gen.Diag)
        write_row('Ndg', orig.Ndg, gen.Ndg)
        write_row('Dist', orig.Dist, gen.Dist)
        write_row('Band', orig.Band, gen.Band)
        write_row('Profil', orig.Profil, gen.Profil)
        write_row('Rmx', orig.Rmx, gen.Rmx)
        write_row('Rmi', orig.Rmi, gen.Rmi)
        write_row('Rstd', orig.Rstd, gen.Rstd)
        write_row('Cmx', orig.Cmx, gen.Cmx)
        write_row('Cmi', orig.Cmi, gen.Cmi)
        write_row('Cstd', orig.Cstd, gen.Cstd)
        
    print(f"\n✓ Saved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare sparse matrix structural features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison with text output
  python matrix_analysis.py original.mtx generated.mtx
  
  # Generate comparison plots
  python matrix_analysis.py original.mtx generated.mtx --plot
  
  # Save plot to specific file
  python matrix_analysis.py original.mtx generated.mtx --plot --plot-output comparison.png
  
  # Specify NNZ scaling power used in generation
  python matrix_analysis.py original.mtx generated.mtx --nnz-power 1.0 --plot
  
  # Just show sparsity pattern of a single matrix
  python matrix_analysis.py matrix.mtx --features-only --plot
"""
    )
    parser.add_argument("original", help="Original matrix file (.mtx)")
    parser.add_argument("generated", nargs='?', help="Generated matrix file (.mtx)")
    parser.add_argument("--method", default="Lanczos", help="Method name for display")
    parser.add_argument("--csv", help="Save comparison to CSV file")
    parser.add_argument("--features-only", action="store_true", 
                       help="Only show features of original matrix")
    parser.add_argument("--nnz-power", type=float, default=1.2,
                       help="NNZ scaling power used in generation (default: 1.2). "
                            "1.0=linear, 2.0=quadratic/density-preserving")
    
    # Visualization arguments
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--plot-output", "-o", type=str, default=None,
                       help="Output path for plot (default: auto-generated)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plot window (just save)")
    parser.add_argument("--dpi", type=int, default=150,
                       help="DPI for saved plots (default: 150)")
    
    args = parser.parse_args()
    
    if args.features_only or args.generated is None:
        # Analyze single matrix
        print(f"Loading {args.original}...")
        matrix = load_matrix_market(args.original)
        features = compute_features(matrix)
        
        print("\n" + "="*110)
        print_header()
        print(f"{'Matrix':<12} {features}")
        print("="*110)
        
        # Plot single matrix if requested
        if args.plot and HAS_MATPLOTLIB:
            plot_path = args.plot_output or f"sparsity_{Path(args.original).stem}.png"
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_sparsity_pattern(matrix, ax=ax, title=Path(args.original).stem)
            fig.savefig(plot_path, dpi=args.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"\n✓ Saved sparsity pattern to {plot_path}")
            if not args.no_show:
                plt.show()
    else:
        # Compare two matrices
        analyze_files(
            args.original, 
            args.generated, 
            args.method, 
            args.nnz_power,
            plot=args.plot,
            plot_path=args.plot_output,
            show_plot=not args.no_show
        )
        
        if args.csv:
            orig_matrix = load_matrix_market(args.original)
            gen_matrix = load_matrix_market(args.generated)
            orig_features = compute_features(orig_matrix)
            gen_features = compute_features(gen_matrix)
            save_comparison_csv(orig_features, gen_features, args.output)


if __name__ == "__main__":
    main()
