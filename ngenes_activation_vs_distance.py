"""Per-sample activation against distance to the tumor centre.

One point per normal sample, drawn twice:

    (distance to the centre of the tumors, number of activated N-genes)
    (distance to the centre of the tumors, number of activated pathways)

A pathway counts as activated in a sample when its frequency there is above
zero, i.e. at least one of its N-genes fired in that sample. Both plots carry
exactly the same samples, so they can be read side by side.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from scipy.stats import gmean

from utils import load_tcga_data, load_pathways, binarize_data
from ngenes import get_ngenes
from tgenes import get_tgenes
# The pathway centre and the distance are defined once, in the N-genes pipeline
from ngenes_analysis import (_tumor_center, _pathway_positions,
                             _distance_components, _distance_to_center)


TCGA_FOLDER = Path('../TCGA-DATA')
EXT_FOLDER = Path('data_external')
OUT_FOLDER = Path('figures and tables/tmp')


def _tumor_center_genes(data, normal_mask):
    """Centroid of the tumor samples in gene space.

    One coordinate per T-gene: the fraction of tumor samples in which that
    gene is deregulated. This is the gene-space counterpart of the pathway
    centre, and it is what the activated-genes plot is measured against.
    """
    (tdata, tgenes, tgenes_above, tgenes_bellow,
     tgenes_max_above, tgenes_min_bellow, *_) = get_tgenes(data, normal_mask)

    bin_tdata = binarize_data(tdata, tgenes_above, tgenes_bellow,
                              tgenes_max_above, tgenes_min_bellow)

    print(f"Tumor center in gene space: {bin_tdata.shape[0]} samples "
          f"over {len(tgenes)} T-genes")

    return bin_tdata.mean()


def _gene_distance(bin_ndata, center):
    """Euclidean distance of every normal sample to the tumor centre, in gene space.

    Same construction as the pathway distance, with genes as the axes: the
    two gene sets are unioned, and a gene missing on one side is a coordinate
    of 0 there rather than a missing value.
    """
    axes = bin_ndata.columns.union(center.index)
    shared = len(bin_ndata.columns.intersection(center.index))
    print(f"Gene axes: {len(axes)} total, {shared} shared by T-genes and N-genes")

    positions = bin_ndata.reindex(columns=axes).fillna(0.0)
    center = center.reindex(axes).fillna(0.0)

    return np.sqrt(((positions - center) ** 2).sum(axis=1))


def _log_fold_change(data, normal_mask):
    """Log fold change of every normal sample against the tumor reference.

    The reference is the per-gene geometric mean of the tumor samples, over
    every gene rather than only the N-genes. Because the geometric mean is the
    exponential of the mean log, the tumors average to exactly zero in this
    space: the centre of the tumor cloud is the origin, so a normal sample's
    distance to it is simply the norm of its own fold-change vector.
    """
    normal = data[normal_mask] + 0.1
    tumor = data[~normal_mask] + 0.1

    ref = gmean(tumor)
    fc = np.log2(normal / ref)

    print(f"Log fold change over all genes: {fc.shape[0]} normal samples, "
          f"{fc.shape[1]} genes, reference from {tumor.shape[0]} tumors")

    return fc


def _center_gap(center_a, center_b):
    """Euclidean distance between two centres defined over different axes.

    Same convention as the sample distances: the axes are unioned and an axis
    missing on one side counts as a coordinate of 0 there.
    """
    axes = center_a.index.union(center_b.index)
    a = center_a.reindex(axes).fillna(0.0)
    b = center_b.reindex(axes).fillna(0.0)

    return float(np.sqrt(((a - b) ** 2).sum()))


def _sample_activation(data_path):
    """Distances, activated N-genes and activated pathways, one row per sample."""
    data, normal_mask = load_tcga_data(data_path)
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} genes")

    (ndata, ngenes, ngenes_above, ngenes_bellow,
     ngenes_max_above, ngenes_min_bellow, *_) = get_ngenes(data, normal_mask)

    print(f"N-genes identified: {len(ngenes)} total")
    print(f"Tumor samples: {(~normal_mask).sum()}, Normal samples: {normal_mask.sum()}")

    pathways_all = load_pathways(EXT_FOLDER / 'pathways/All_pathways.csv')

    bin_ndata = binarize_data(ndata, ngenes_above, ngenes_bellow,
                              ngenes_max_above, ngenes_min_bellow)

    # Gene space: position is the 0/1 deregulation vector of the sample, the
    # centre is the tumors averaged over the T-genes.
    gene_center = _tumor_center_genes(data, normal_mask)
    gene_distance = _gene_distance(bin_ndata, gene_center)

    # Log fold change space: every gene, N-gene or not. The tumor centre is
    # the origin here, so the distance is the norm of the sample's own vector.
    fc = _log_fold_change(data, normal_mask)
    all_genes_distance = np.sqrt((fc ** 2).sum(axis=1))

    # Pathway space: position is the per-pathway frequency of the sample, the
    # centre is the tumors averaged over the T-gene pathways.
    center = _tumor_center(data, normal_mask, pathways_all)
    pw_normal = _pathway_positions(ndata, ngenes_above, ngenes_bellow,
                                   ngenes_max_above, ngenes_min_bellow,
                                   ngenes, pathways_all)
    components = _distance_components(pw_normal, center)
    pathway_distance = _distance_to_center(components)['distance']

    # Distance between the normal and tumor centres, in each of the three
    # spaces, so every plot's x axis has its own reference value.
    print("\nDistance from the normal centre to the tumor centre:")
    print(f"  gene space             : {_center_gap(bin_ndata.mean(), gene_center):.3f}")
    # the tumor centre is the origin of the fold change space
    print(f"  all-gene log fold change: {np.sqrt((fc.mean() ** 2).sum()):.3f}")
    print(f"  pathway space          : {_center_gap(pw_normal.mean(), center):.3f}\n")

    # Activated N-genes per sample, counted over every N-gene rather than only
    # those that happen to sit in a pathway.
    active_ngenes = bin_ndata.sum(axis=1)

    # Activated pathways per sample: frequency > 0 means at least one N-gene
    # of that pathway fired in the sample.
    active_pathways = (pw_normal > 0).sum(axis=1)

    # Built positionally: TCGA sample ids repeat, and aligning Series on a
    # duplicated index would either raise or silently cross-join the rows.
    # All four come from ndata in the same row order.
    return pd.DataFrame({
        'gene_distance': gene_distance.to_numpy(),
        'all_genes_distance': all_genes_distance.to_numpy(),
        'pathway_distance': pathway_distance.to_numpy(),
        'active_ngenes': active_ngenes.to_numpy(),
        'active_pathways': active_pathways.to_numpy(),
    }, index=pw_normal.index), len(ngenes), pw_normal.shape[1]


def _plot_activation(df, distance_column, column, total, axis_label, space,
                     tissue, output_dir, filename, color):
    """Scatter one activation count against the distance to the tumor centre.

    ``space`` names the space the distance is measured in, which has to match
    what is being counted: genes against the gene-space distance, pathways
    against the pathway-space one.
    """
    counts = df[column]
    distance = df[distance_column]

    print(f"{axis_label} vs {space} distance")

    hover_text = [f"<b>{sample}</b><br>"
                  f"Distance to tumor centre ({space} space): {d:.3f}<br>"
                  f"{axis_label}: {c} of {total} ({c / total:.1%})"
                  for sample, d, c in zip(df.index, distance, counts)]

    fig = go.Figure(go.Scattergl(
        x=distance.to_numpy(),
        y=counts.to_numpy(),
        mode='markers',
        marker=dict(size=6, color=color, opacity=0.65),
        hovertext=hover_text,
        hoverinfo='text',
    ))

    fig.update_layout(
        title=f'{axis_label} vs distance to tumor centre - {tissue}'
              f'<br><sub>{len(df)} normal samples, distance in {space} space</sub>',
        xaxis_title=f'Distance to the centre of the tumor samples ({space} space)',
        yaxis_title=f'{axis_label} (of {total})',
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=700,
        showlegend=False,
    )

    fig.write_html(output_dir / f'{filename}.html')
    print(f"Saved: {filename}.html")

    fig.write_image(output_dir / f'{filename}.pdf')
    print(f"Saved: {filename}.pdf")

    fig.show()


def analyze_activation_vs_distance(tissue: str, tissue_folder: str):
    """Both activation plots for one tissue."""
    output_dir = OUT_FOLDER / (tissue + '_ngenes')
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Analyzing activation vs distance for tissue: {tissue}")

    df, total_ngenes, total_pathways = _sample_activation(TCGA_FOLDER / tissue_folder)

    df.to_csv(output_dir / 'activation_vs_distance.csv')
    print(f"Saved: activation_vs_distance.csv ({len(df)} normal samples)\n")

    _plot_activation(df, 'gene_distance', 'active_ngenes', total_ngenes,
                     'Activated N-genes', 'gene',
                     tissue, output_dir, 'distance_vs_active_ngenes', '#2CA02C')

    _plot_activation(df, 'all_genes_distance', 'active_ngenes', total_ngenes,
                     'Activated N-genes', 'all-gene log fold change',
                     tissue, output_dir, 'distance_vs_active_ngenes_all_genes', '#FF7F0E')

    _plot_activation(df, 'pathway_distance', 'active_pathways', total_pathways,
                     'Activated pathways', 'pathway',
                     tissue, output_dir, 'distance_vs_active_pathways', '#9467BD')

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    analyze_activation_vs_distance('PRAD', '2. TCGA-PRAD')
    # analyze_activation_vs_distance('GBM', '6. TCGA-GBM')
