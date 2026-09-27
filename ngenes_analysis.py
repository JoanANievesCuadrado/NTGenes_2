"""N-genes analysis pipeline."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path

from utils import (load_tcga_data, load_pathways, intersect_data_pathway, binarize_data,
                   ge2pe_fast, get_row_genes, load_ensembl_symbol_map)
from ngenes import get_ngenes
from tgenes import get_tgenes


TCGA_FOLDER = Path('../TCGA-DATA')
EXT_FOLDER = Path('data_external')
OUT_FOLDER = Path('figures and tables')


def _load_and_prepare_ngenes(data_path):
    """Load data and identify N-genes."""
    data, normal_mask = load_tcga_data(data_path)
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} genes")

    ndata, ngenes, ngenes_above, ngenes_bellow, ngenes_max_above, ngenes_min_bellow, ngenes_above_only, ngenes_outside, ngenes_bellow_only = get_ngenes(data, normal_mask)

    print(f"N-genes identified: {len(ngenes)} total")
    print(f"  - Above only (na): {len(ngenes_above_only)}")
    print(f"  - Outside (no): {len(ngenes_outside)}")
    print(f"  - Below only (nb): {len(ngenes_bellow_only)}")
    print(f"Tumor samples: {(~normal_mask).sum()}, Normal samples: {normal_mask.sum()}")

    return data, normal_mask, ndata, ngenes, ngenes_above, ngenes_bellow, ngenes_max_above, ngenes_min_bellow


def _prepare_pathway_data(bin_ndata, ngenes):
    """Load pathways and intersect with N-genes data."""
    pathways_all = load_pathways(EXT_FOLDER / 'pathways/All_pathways.csv')
    pathways = pathways_all.copy()
    bin_ndata, pathways, common_genes = intersect_data_pathway(bin_ndata, pathways, ngenes)

    print(f"Common genes in pathways: {len(common_genes)}")
    print(f"Unique pathways: {pathways.Pathway.nunique()}")

    return pathways_all, pathways, bin_ndata


def _calculate_pathway_statistics(pathways_all, pathways, bin_ndata, gene_labels):
    """Mean pathway frequency over normal samples.

    For one sample s and one pathway p the frequency is

        freq[s, p] = (N-genes of p activated in s) / (N-genes of p)

    and the value reported per pathway is the mean of freq[:, p] over all
    normal samples.
    """
    # samples x pathways matrix of per-sample frequencies, then average
    pw_freq = ge2pe_fast(bin_ndata, pathways, mode='proportion')
    mean_freq = pw_freq.mean()

    # Spread of the per-sample frequencies; undefined for a single sample
    std_freq = pw_freq.std().reindex(mean_freq.index).fillna(0.0)

    pathway_names = mean_freq.index.tolist()

    # Per-gene deregulation fraction, used for the top-genes hover list
    bin_data_mean = bin_ndata.mean()

    # Precomputed lookups so the loop does not rescan the pathway table
    pathway_sizes = pathways_all.Pathway.value_counts()
    pw_genes_map = (pathways.drop_duplicates(subset=['Gene', 'Pathway'])
                            .groupby('Pathway').Gene.apply(np.array))

    total_genes = []
    ngenes_count = []
    mean_active = []
    top_genes_list = []

    for pw in pathway_names:
        total_genes.append(pathway_sizes.get(pw, 0))

        pw_genes = pw_genes_map[pw]
        ngenes_count.append(len(pw_genes))

        # Numerator of the frequency: activated N-genes averaged over samples
        mean_active.append(mean_freq[pw] * len(pw_genes))

        # Label with gene symbols, falling back to the ensembl ID when unmapped
        top = bin_data_mean[pw_genes].sort_values(ascending=False).head(10)
        top.index = [gene_labels.get(g, g) for g in top.index]
        top_genes_list.append(top)

    return pathway_names, mean_freq.to_numpy(), std_freq.to_numpy(), total_genes, ngenes_count, mean_active, top_genes_list


def _pathway_positions(sample_data, genes_above, genes_bellow, max_above, min_bellow,
                       genes, pathways_all):
    """Embed samples in pathway-frequency space.

    Returns a samples x pathways frame where entry [s, p] is the fraction of
    the deregulated genes of pathway p that are active in sample s.
    """
    bin_data = binarize_data(sample_data, genes_above, genes_bellow, max_above, min_bellow)
    bin_data, pathways, _ = intersect_data_pathway(bin_data, pathways_all.copy(), genes)

    return ge2pe_fast(bin_data, pathways, mode='proportion')


def _tumor_center(data, normal_mask, pathways_all):
    """Centroid of the tumor samples in T-gene pathway-frequency space.

    The tumor coordinates have to come from the T-genes: those are the genes
    whose tumor expression leaves the normal range, so tumor samples take
    non-zero values along them.
    """
    (tdata, tgenes, tgenes_above, tgenes_bellow,
     tgenes_max_above, tgenes_min_bellow, *_) = get_tgenes(data, normal_mask)

    pw_tumor = _pathway_positions(tdata, tgenes_above, tgenes_bellow,
                                  tgenes_max_above, tgenes_min_bellow,
                                  tgenes, pathways_all)

    print(f"Tumor center: {pw_tumor.shape[0]} samples over {pw_tumor.shape[1]} pathways "
          f"({len(tgenes)} T-genes)")

    return pw_tumor.mean()


def _distance_components(pw_normal, center):
    """Per-pathway distance of every normal sample from the tumor centre.

    Returns a samples x pathways frame holding |position - centre|, the
    contribution each pathway makes to that sample's total distance.

    Normal positions and the centre are defined over different gene sets, so
    the pathway axes are unioned. A pathway missing on one side has no
    deregulated gene there, which is a coordinate of 0, not a missing value.
    """
    axes = pw_normal.columns.union(center.index)
    shared = len(pw_normal.columns.intersection(center.index))
    print(f"Pathway axes: {len(axes)} total, {shared} shared by T-genes and N-genes")

    positions = pw_normal.reindex(columns=axes).fillna(0.0)
    center = center.reindex(axes).fillna(0.0)

    return (positions - center).abs()


def _distance_to_center(components):
    """Total Euclidean distance of every normal sample to the tumor centre."""
    distances = np.sqrt((components ** 2).sum(axis=1))

    return pd.DataFrame({
        'distance': distances,
        # same distance expressed as a typical per-pathway gap
        'rms_per_pathway': distances / np.sqrt(components.shape[1]),
    })


def _plot_distance_components(components, tissue, output_dir):
    """Per-pathway distance of each normal sample, pathways sorted by the mean.

    One row per pathway, ordered by the mean distance across normal samples,
    with one marker per normal sample plus the mean itself.
    """
    mean_component = components.mean()

    # Spread of the per-pathway distance across normal samples
    std_component = components.std().reindex(mean_component.index).fillna(0.0)

    order = mean_component.sort_values().index  # ascending
    components = components[order]
    mean_sorted = mean_component[order]
    std_sorted = std_component[order]
    ranks = np.arange(len(order))

    # Both coordinates are fractions in [0, 1], so |position - centre| is too
    upper = np.clip(mean_sorted + std_sorted, 0.0, 1.0)
    lower = np.clip(mean_sorted - std_sorted, 0.0, 1.0)

    fig = go.Figure()

    # Band first so the markers and the mean curve draw on top of it. The
    # pathway runs along y here, so the band spans x at each rank.
    fig.add_trace(go.Scattergl(
        x=np.concatenate([upper.to_numpy(), lower.to_numpy()[::-1]]),
        y=np.concatenate([ranks, ranks[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 0, 0.12)',
        line=dict(width=0),
        hoverinfo='skip',
        name='± 1 std',
    ))

    for sample in components.index:
        values = components.loc[sample]
        fig.add_trace(go.Scattergl(
            x=values.to_numpy(),
            y=ranks,
            mode='markers',
            marker=dict(size=4, opacity=0.45),
            name=str(sample),
            hovertext=[f"<b>{pw}</b><br>Sample: {sample}<br>"
                       f"Distance: {v:.3f}<br>Mean over samples: {m:.3f} ± {sd:.3f}"
                       for pw, v, m, sd in zip(order, values, mean_sorted, std_sorted)],
            hoverinfo='text',
        ))

    fig.add_trace(go.Scattergl(
        x=mean_sorted.to_numpy(),
        y=ranks,
        mode='lines',
        line=dict(color='black', width=1.5),
        name='Mean',
        hoverinfo='skip',
    ))

    fig.update_layout(
        title=f'Normal samples: per-pathway distance to tumor centre - {tissue}',
        xaxis_title='Distance to tumor centre in that pathway',
        yaxis_title='Pathways (sorted by mean distance, ascending)',
        hovermode='closest',
        template='plotly_white',
        width=1100,
        height=900,
        showlegend=False,
    )

    fig.write_html(output_dir / 'normal_distance_components.html')
    print("Saved: normal_distance_components.html")

    fig.write_image(output_dir / 'normal_distance_components.pdf')
    print("Saved: normal_distance_components.pdf")

    fig.show()


def _sort_pathways_by_frequency(pathway_names, frequencies, stds, total_genes, ngenes_count, mean_active, top_genes_list):
    """Sort pathways by mean frequency in descending order."""
    args = np.argsort(-frequencies)
    sorted_pathways = [pathway_names[i] for i in args]
    sorted_freq = frequencies[args]
    sorted_std = stds[args]
    sorted_total = [total_genes[i] for i in args]
    sorted_ngenes = [ngenes_count[i] for i in args]
    sorted_active = [mean_active[i] for i in args]
    sorted_genes = [top_genes_list[i] for i in args]

    return sorted_pathways, sorted_freq, sorted_std, sorted_total, sorted_ngenes, sorted_active, sorted_genes


def _create_hover_text(sorted_pathways, sorted_freq, sorted_std, sorted_total, sorted_ngenes, sorted_active, sorted_genes):
    """Create hover text for Plotly visualization."""
    hover_text = []
    for pw, freq, std, tot, ng, act, genes in zip(sorted_pathways, sorted_freq, sorted_std, sorted_total, sorted_ngenes, sorted_active, sorted_genes):
        genes_str = "<br>".join([f"  • {gene}: {value:.2f}" for gene, value in genes.items()])
        coverage = ng / tot if tot > 0 else 0

        # Only call it a "top" slice when the list is actually truncated
        n_shown = len(genes)
        header = (f"Top {n_shown} of {ng} N-genes" if n_shown < ng else "N-genes")

        text = (f"<b>{pw}</b><br>" +
                f"Mean frequency: {freq:.3f} ± {std:.3f}<br>" +
                f"Active N-genes per sample: {act:.2f} of {ng}<br>" +
                f"Pathway size: {tot}<br>" +
                f"N-genes in pathway: {ng} ({coverage:.1%} of pathway)<br>" +
                f"<b>{header} (fraction of samples deregulated):</b><br>" +
                genes_str)
        hover_text.append(text)
    return hover_text


def _create_and_save_plot(sorted_pathways, sorted_freq, sorted_std, hover_text, tissue, output_dir):
    """Create Plotly figure and save as HTML and PDF."""
    x = list(range(len(sorted_pathways)))

    # A frequency is a fraction of the pathway's N-genes, so the band cannot
    # leave [0, 1] however wide the standard deviation is.
    upper = np.clip(sorted_freq + sorted_std, 0.0, 1.0)
    lower = np.clip(sorted_freq - sorted_std, 0.0, 1.0)

    fig = go.Figure()

    # Band first so the mean curve draws on top of it
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(upper) + list(lower[::-1]),
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(width=0),
        hoverinfo='skip',
        name='± 1 std',
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=sorted_freq,
        mode='lines+markers',
        marker=dict(size=6, color='#2CA02C'),
        line=dict(color='#2CA02C', width=2),
        hovertext=hover_text,
        hoverinfo='text',
        name='N-Genes'
    ))

    fig.update_layout(
        title=f'N-Genes Pathway Expression - {tissue}',
        xaxis_title='Pathways (sorted by mean frequency)',
        yaxis_title='Mean fraction of pathway N-genes activated per sample',
        hovermode='closest',
        template='plotly_white',
        width=1400,
        height=600,
        showlegend=False
    )

    fig.write_html(output_dir / 'pathway_distribution.html')
    print("Saved: pathway_distribution.html")

    fig.write_image(output_dir / 'pathway_distribution.pdf')
    print("Saved: pathway_distribution.pdf")

    fig.show()


def analyze_ngenes(tissue: str, tissue_folder: str):
    """Complete N-genes analysis pipeline."""
    output_dir = OUT_FOLDER / (tissue + '_ngenes')
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Analyzing N-genes for tissue: {tissue}")

    # Load and prepare N-genes
    data_path = TCGA_FOLDER / tissue_folder
    data, normal_mask, ndata, ngenes, ngenes_above, ngenes_bellow, ngenes_max_above, ngenes_min_bellow = _load_and_prepare_ngenes(data_path)

    # Binarize data
    bin_ndata = binarize_data(ndata, ngenes_above, ngenes_bellow, ngenes_max_above, ngenes_min_bellow)
    print("Data binarized")

    # Prepare pathway data
    pathways_all, pathways, bin_ndata = _prepare_pathway_data(bin_ndata, ngenes)

    # Distance of every normal sample to the centre of the tumor cloud.
    # The centre comes from the T-genes, the normal positions from the N-genes.
    center = _tumor_center(data, normal_mask, pathways_all)
    pw_normal = _pathway_positions(ndata, ngenes_above, ngenes_bellow,
                                   ngenes_max_above, ngenes_min_bellow,
                                   ngenes, pathways_all)
    components = _distance_components(pw_normal, center)

    distances = _distance_to_center(components)
    distances.to_csv(output_dir / 'normal_distance_to_tumor_center.csv')
    print(f"Saved: normal_distance_to_tumor_center.csv\n{distances.sort_values('distance')}\n")

    _plot_distance_components(components, tissue, output_dir)

    # Gene symbols for the hover labels
    gene_labels = load_ensembl_symbol_map(get_row_genes(TCGA_FOLDER / 'rows_genes2.xlsx'))
    mapped = sum(g in gene_labels for g in bin_ndata.columns)
    print(f"Gene symbols mapped: {mapped} of {bin_ndata.shape[1]}")

    # Calculate pathway statistics
    pathway_names, frequencies, stds, total_genes, ngenes_count, mean_active, top_genes_list = _calculate_pathway_statistics(pathways_all, pathways, bin_ndata, gene_labels)

    # Sort and prepare for visualization
    sorted_pathways, sorted_freq, sorted_std, sorted_total, sorted_ngenes, sorted_active, sorted_genes = _sort_pathways_by_frequency(pathway_names, frequencies, stds, total_genes, ngenes_count, mean_active, top_genes_list)

    # Create hover text
    hover_text = _create_hover_text(sorted_pathways, sorted_freq, sorted_std, sorted_total, sorted_ngenes, sorted_active, sorted_genes)

    # Create and save plot
    _create_and_save_plot(sorted_pathways, sorted_freq, sorted_std, hover_text, tissue, output_dir)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    analyze_ngenes('GBM', '6. TCGA-GBM')
    # analyze_ngenes('PRAD', '2. TCGA-PRAD')
