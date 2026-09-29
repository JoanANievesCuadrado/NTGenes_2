"""Per-sample activation against distance to the tumor centre.

One point per normal sample, drawn twice:

    (distance to the centre of the tumors, number of activated N-genes)
    (distance to the centre of the tumors, number of activated pathways)

A pathway counts as activated in a sample when its frequency there is above
zero, i.e. at least one of its N-genes fired in that sample. Both plots carry
exactly the same samples, so they can be read side by side.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch

from matplotlib.colors import to_hex
from pathlib import Path
from scipy.stats import gmean

from utils import (load_tcga_data, load_pathways, binarize_data,
                   intersect_data_pathway, get_row_genes, load_ensembl_symbol_map)
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
     ngenes_max_above, ngenes_min_bellow,
     ngenes_above_only, ngenes_outside, ngenes_bellow_only) = get_ngenes(data, normal_mask)

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

    # Which pathway holds which N-gene, for the per-sample exports. The same
    # intersection _pathway_positions performs internally, so the membership
    # matches the frequencies exactly.
    _, pathways_n, _ = intersect_data_pathway(bin_ndata, pathways_all.copy(), ngenes)
    pw_genes_map = (pathways_n.drop_duplicates(subset=['Gene', 'Pathway'])
                              .groupby('Pathway').Gene.apply(list))

    context = {
        'bin_data': bin_ndata,
        'pw_data': pw_normal,
        'cohort_fraction': bin_ndata.mean(),
        'pw_genes_map': pw_genes_map,
        'pathway_sizes': pathways_all.Pathway.value_counts(),
        'gene_class': {**{g: 'na' for g in ngenes_above_only},
                       **{g: 'no' for g in ngenes_outside},
                       **{g: 'nb' for g in ngenes_bellow_only}},
    }

    # Built positionally: TCGA sample ids repeat, and aligning Series on a
    # duplicated index would either raise or silently cross-join the rows.
    # All four come from ndata in the same row order.
    return pd.DataFrame({
        'gene_distance': gene_distance.to_numpy(),
        'all_genes_distance': all_genes_distance.to_numpy(),
        'pathway_distance': pathway_distance.to_numpy(),
        'active_ngenes': active_ngenes.to_numpy(),
        'active_pathways': active_pathways.to_numpy(),
    }, index=pw_normal.index), len(ngenes), pw_normal.shape[1], context


def _dendrogram(pw_data, tissue, output_dir, color_threshold=None):
    """Ward dendrogram over the pathway vector of every sample.

    Returns the colour scipy gave each sample, ordered by the rows of
    ``pw_data``. The dendrogram reports its leaves in drawing order, so
    ``leaves[k]`` names the sample drawn at position k and
    ``leaves_color_list[k]`` is that leaf's colour; the two have to be zipped
    back together to recover a per-sample colour.
    """
    linkage_matrix = sch.linkage(pw_data, method='ward')

    fig_height = 4
    fig_width = fig_height * 1.618
    fontsize = 11.5

    plt.figure(figsize=(fig_width, fig_height))
    dendro = sch.dendrogram(linkage_matrix, no_labels=True,
                            color_threshold=color_threshold)
    plt.ylabel('Distances', fontsize=fontsize)
    plt.xlabel('Samples', fontsize=fontsize)
    plt.title(tissue)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / 'pathway_dendrogram.pdf')
    plt.close()
    print("Saved: pathway_dendrogram.pdf")

    # scipy names colours through the matplotlib cycle ("C0", "C1", ...), which
    # plotly cannot read, so resolve them to hex here.
    colors = np.empty(pw_data.shape[0], dtype=object)
    for position, sample in enumerate(dendro['leaves']):
        colors[sample] = to_hex(dendro['leaves_color_list'][position])

    counts = pd.Series(colors).value_counts()
    print(f"Dendrogram: {len(counts)} leaf colours over {len(colors)} samples "
          f"({', '.join(f'{c}:{n}' for c, n in counts.items())})")

    return colors


def _save_gene_frequency(context, gene_labels, output_dir, filename):
    """Activation frequency of every N-gene across the normal samples.

    The frequency is the fraction of normal samples in which the gene is
    deregulated, so it cannot fall below the threshold that qualified the gene
    as an N-gene in the first place.
    """
    bin_data = context['bin_data']
    frequency = context['cohort_fraction']
    gene_class = context['gene_class']

    genes = list(frequency.index)
    table = pd.DataFrame({
        'gene': [gene_labels.get(g, g) for g in genes],
        'class': [gene_class.get(g, '') for g in genes],
        'frequency': frequency.to_numpy().round(4),
        'n_samples_active': bin_data.sum().to_numpy(),
        'n_samples': bin_data.shape[0],
    }).sort_values('frequency', ascending=False)

    table.to_csv(output_dir / filename, index=False)
    print(f"Saved: {filename} ({len(table)} genes, "
          f"frequency {table.frequency.min():.3f} to {table.frequency.max():.3f})")


def _save_least_active(df, count_column, context, gene_labels, output_dir, n=10):
    """Write a genes file and a pathways file for each least active sample.

    Takes the ``n`` samples with the fewest active genes and the ``n`` with the
    fewest active pathways, then writes their detail into
    ``least_active_samples/`` together with a summary row per sample. The two
    selections usually overlap, so their union is used and each sample records
    the criteria that picked it.

    Everything is addressed positionally: TCGA sample ids repeat, so a label
    lookup would pull several samples at once.
    """
    by_genes = set(np.argsort(df[count_column].to_numpy(), kind='stable')[:n].tolist())
    by_pathways = set(np.argsort(df['active_pathways'].to_numpy(), kind='stable')[:n].tolist())
    selected = sorted(by_genes | by_pathways)

    folder = output_dir / 'least_active_samples'
    folder.mkdir(exist_ok=True, parents=True)

    print(f"Least active samples: {len(by_genes)} by genes, {len(by_pathways)} by "
          f"pathways, {len(selected)} together")

    bin_data = context['bin_data']
    pw_data = context['pw_data']
    cohort_fraction = context['cohort_fraction']
    gene_class = context['gene_class']
    pw_genes_map = context['pw_genes_map']
    pathway_sizes = context['pathway_sizes']

    repeated = df.index.duplicated(keep=False)
    summary = []

    for i in selected:
        sample = df.index[i]

        # Sample ids repeat in TCGA, so a shared id would have the two samples
        # overwriting each other's files; disambiguate by row position.
        stem = f'{sample}_{i}' if repeated[i] else str(sample)
        stem = re.sub(r'[^A-Za-z0-9._-]', '_', stem)

        if i in by_genes and i in by_pathways:
            reason = 'genes+pathways'
        elif i in by_genes:
            reason = 'genes'
        else:
            reason = 'pathways'

        # --- genes ----------------------------------------------------------
        genes_row = bin_data.iloc[i]
        active_genes = list(genes_row.index[genes_row > 0])

        genes_df = pd.DataFrame({
            'gene': [gene_labels.get(g, g) for g in active_genes],
            'class': [gene_class.get(g, '') for g in active_genes],
            'cohort_fraction': [round(float(cohort_fraction[g]), 4) for g in active_genes],
        }).sort_values('cohort_fraction', ascending=False)
        genes_df.to_csv(folder / f'{stem}_genes.csv', index=False)

        # --- pathways -------------------------------------------------------
        active = set(active_genes)
        pw_row = pw_data.iloc[i]

        records = []
        for pathway in pw_row.index[pw_row > 0]:
            members = pw_genes_map[pathway]
            hits = [g for g in members if g in active]
            records.append({
                'pathway': pathway,
                'frequency': round(float(pw_row[pathway]), 4),
                'active_genes': len(hits),
                'genes_in_pathway': len(members),
                'pathway_size': int(pathway_sizes.get(pathway, 0)),
                'gene_names': ';'.join(gene_labels.get(g, g) for g in hits),
            })

        pw_df = pd.DataFrame(records).sort_values('frequency', ascending=False)
        pw_df.to_csv(folder / f'{stem}_pathways.csv', index=False)

        summary.append({
            'sample': sample,
            'file_stem': stem,
            'selected_by': reason,
            'n_active_genes': df[count_column].iloc[i],
            'n_active_pathways': df['active_pathways'].iloc[i],
            'gene_distance': round(float(df['gene_distance'].iloc[i]), 4),
            'all_genes_distance': round(float(df['all_genes_distance'].iloc[i]), 4),
            'pathway_distance': round(float(df['pathway_distance'].iloc[i]), 4),
        })

    pd.DataFrame(summary).to_csv(folder / 'summary.csv', index=False)
    print(f"Saved: least_active_samples/ ({2 * len(selected)} files plus summary.csv)")


def _plot_activation(df, distance_column, column, total, axis_label, space,
                     tissue, output_dir, filename, color, log_y=False):
    """Scatter one activation count against the distance to the tumor centre.

    ``space`` names the space the distance is measured in, which has to match
    what is being counted: genes against the gene-space distance, pathways
    against the pathway-space one.
    """
    counts = df[column]
    distance = df[distance_column]

    print(f"{axis_label} vs {space} distance")

    # A log axis simply drops non-positive values, so say so rather than let
    # samples disappear from the figure without a trace.
    if log_y and (counts <= 0).any():
        print(f"  warning: {(counts <= 0).sum()} samples have {axis_label} = 0 "
              f"and cannot be shown on a log axis")

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
        yaxis_type='log' if log_y else 'linear',
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

    df, total_ngenes, total_pathways, context = _sample_activation(
        TCGA_FOLDER / tissue_folder)

    df.to_csv(output_dir / 'activation_vs_distance.csv')
    print(f"Saved: activation_vs_distance.csv ({len(df)} normal samples)\n")

    gene_labels = load_ensembl_symbol_map(get_row_genes(TCGA_FOLDER / 'rows_genes2.xlsx'))
    _save_gene_frequency(context, gene_labels, output_dir, 'ngene_activation_frequency.csv')
    _save_least_active(df, 'active_ngenes', context, gene_labels, output_dir)
    print()

    # Cluster the samples on their pathway vectors; the leaf colours carry over
    # to the two fold-change plots so the same sample keeps the same colour.
    cluster_colors = _dendrogram(context['pw_data'], tissue, output_dir)

    _plot_activation(df, 'gene_distance', 'active_ngenes', total_ngenes,
                     'Activated N-genes', 'gene',
                     tissue, output_dir, 'distance_vs_active_ngenes', '#2CA02C')

    _plot_activation(df, 'all_genes_distance', 'active_ngenes', total_ngenes,
                     'Activated N-genes', 'all-gene log fold change',
                     tissue, output_dir, 'distance_vs_active_ngenes_all_genes',
                     cluster_colors, log_y=True)

    _plot_activation(df, 'all_genes_distance', 'active_pathways', total_pathways,
                     'Activated pathways', 'all-gene log fold change',
                     tissue, output_dir, 'distance_vs_active_pathways_all_genes',
                     cluster_colors, log_y=True)

    _plot_activation(df, 'pathway_distance', 'active_pathways', total_pathways,
                     'Activated pathways', 'pathway',
                     tissue, output_dir, 'distance_vs_active_pathways', '#9467BD')

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    analyze_activation_vs_distance('PRAD', '2. TCGA-PRAD')
    # analyze_activation_vs_distance('GBM', '6. TCGA-GBM')
