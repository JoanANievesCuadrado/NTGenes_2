"""T-genes analysis pipeline."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path

from utils import (load_tcga_data, load_pathways, intersect_data_pathway, binarize_data,
                   ge2pe_fast, get_row_genes, load_ensembl_symbol_map)
from tgenes import get_tgenes


TCGA_FOLDER = Path('../TCGA-DATA')
EXT_FOLDER = Path('data_external')
OUT_FOLDER = Path('figures and tables')


def _load_and_prepare_tgenes(data_path):
    """Load data and identify T-genes."""
    data, normal_mask = load_tcga_data(data_path)
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} genes")

    tdata, tgenes, tgenes_above, tgenes_bellow, tgenes_max_above, tgenes_min_bellow, tgenes_above_only, tgenes_outside, tgenes_bellow_only = get_tgenes(data, normal_mask)

    print(f"T-genes identified: {len(tgenes)} total")
    print(f"  - Above only (ta): {len(tgenes_above_only)}")
    print(f"  - Outside (to): {len(tgenes_outside)}")
    print(f"  - Below only (tb): {len(tgenes_bellow_only)}")
    print(f"Tumor samples: {(~normal_mask).sum()}, Normal samples: {normal_mask.sum()}")

    return tdata, tgenes, tgenes_above, tgenes_bellow, tgenes_max_above, tgenes_min_bellow


def _prepare_pathway_data(bin_tdata, tgenes):
    """Load pathways and intersect with T-genes data."""
    pathways_all = load_pathways(EXT_FOLDER / 'pathways/All_pathways.csv')
    pathways = pathways_all.copy()
    bin_tdata, pathways, common_genes = intersect_data_pathway(bin_tdata, pathways, tgenes)

    print(f"Common genes in pathways: {len(common_genes)}")
    print(f"Unique pathways: {pathways.Pathway.nunique()}")

    return pathways_all, pathways, bin_tdata


def _calculate_pathway_statistics(pathways_all, pathways, bin_tdata, gene_labels):
    """Mean pathway frequency over tumor samples.

    For one sample s and one pathway p the frequency is

        freq[s, p] = (T-genes of p activated in s) / (T-genes of p)

    and the value reported per pathway is the mean of freq[:, p] over all
    tumor samples.
    """
    # samples x pathways matrix of per-sample frequencies, then average
    pw_freq = ge2pe_fast(bin_tdata, pathways, mode='proportion')
    mean_freq = pw_freq.mean()

    pathway_names = mean_freq.index.tolist()

    # Per-gene deregulation fraction, used for the top-genes hover list
    bin_data_mean = bin_tdata.mean()

    # Precomputed lookups so the loop does not rescan the pathway table
    pathway_sizes = pathways_all.Pathway.value_counts()
    pw_genes_map = (pathways.drop_duplicates(subset=['Gene', 'Pathway'])
                            .groupby('Pathway').Gene.apply(np.array))

    total_genes = []
    tgenes_count = []
    mean_active = []
    top_genes_list = []

    for pw in pathway_names:
        total_genes.append(pathway_sizes.get(pw, 0))

        pw_genes = pw_genes_map[pw]
        tgenes_count.append(len(pw_genes))

        # Numerator of the frequency: activated T-genes averaged over samples
        mean_active.append(mean_freq[pw] * len(pw_genes))

        # Label with gene symbols, falling back to the ensembl ID when unmapped
        top = bin_data_mean[pw_genes].sort_values(ascending=False).head(10)
        top.index = [gene_labels.get(g, g) for g in top.index]
        top_genes_list.append(top)

    return pathway_names, mean_freq.to_numpy(), total_genes, tgenes_count, mean_active, top_genes_list


def _sort_pathways_by_frequency(pathway_names, frequencies, total_genes, tgenes_count, mean_active, top_genes_list):
    """Sort pathways by mean frequency in descending order."""
    args = np.argsort(-frequencies)
    sorted_pathways = [pathway_names[i] for i in args]
    sorted_freq = frequencies[args]
    sorted_total = [total_genes[i] for i in args]
    sorted_tgenes = [tgenes_count[i] for i in args]
    sorted_active = [mean_active[i] for i in args]
    sorted_genes = [top_genes_list[i] for i in args]

    return sorted_pathways, sorted_freq, sorted_total, sorted_tgenes, sorted_active, sorted_genes


def _create_hover_text(sorted_pathways, sorted_freq, sorted_total, sorted_tgenes, sorted_active, sorted_genes):
    """Create hover text for Plotly visualization."""
    hover_text = []
    for pw, freq, tot, tg, act, genes in zip(sorted_pathways, sorted_freq, sorted_total, sorted_tgenes, sorted_active, sorted_genes):
        genes_str = "<br>".join([f"  • {gene}: {value:.2f}" for gene, value in genes.items()])
        coverage = tg / tot if tot > 0 else 0

        # Only call it a "top" slice when the list is actually truncated
        n_shown = len(genes)
        header = (f"Top {n_shown} of {tg} T-genes" if n_shown < tg else "T-genes")

        text = (f"<b>{pw}</b><br>" +
                f"Mean frequency: {freq:.3f}<br>" +
                f"Active T-genes per sample: {act:.2f} of {tg}<br>" +
                f"Pathway size: {tot}<br>" +
                f"T-genes in pathway: {tg} ({coverage:.1%} of pathway)<br>" +
                f"<b>{header} (fraction of samples deregulated):</b><br>" +
                genes_str)
        hover_text.append(text)
    return hover_text


def _create_and_save_plot(sorted_pathways, sorted_freq, hover_text, tissue, output_dir):
    """Create Plotly figure and save as HTML and PDF."""
    fig = go.Figure(data=go.Scatter(
        x=list(range(len(sorted_pathways))),
        y=sorted_freq,
        mode='lines+markers',
        marker=dict(size=6, color='#00A6D6'),
        line=dict(color='#00A6D6', width=2),
        hovertext=hover_text,
        hoverinfo='text',
        name='T-Genes'
    ))

    fig.update_layout(
        title=f'T-Genes Pathway Expression - {tissue}',
        xaxis_title='Pathways (sorted by mean frequency)',
        yaxis_title='Mean fraction of pathway T-genes activated per sample',
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


def analyze_tgenes(tissue: str, tissue_folder: str):
    """Complete T-genes analysis pipeline."""
    output_dir = OUT_FOLDER / (tissue + '_tgenes')
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Analyzing T-genes for tissue: {tissue}")

    # Load and prepare T-genes
    data_path = TCGA_FOLDER / tissue_folder
    tdata, tgenes, tgenes_above, tgenes_bellow, tgenes_max_above, tgenes_min_bellow = _load_and_prepare_tgenes(data_path)

    # Binarize data
    bin_tdata = binarize_data(tdata, tgenes_above, tgenes_bellow, tgenes_max_above, tgenes_min_bellow)
    print("Data binarized")

    # Prepare pathway data
    pathways_all, pathways, bin_tdata = _prepare_pathway_data(bin_tdata, tgenes)

    # Gene symbols for the hover labels
    gene_labels = load_ensembl_symbol_map(get_row_genes(TCGA_FOLDER / 'rows_genes2.xlsx'))
    mapped = sum(g in gene_labels for g in bin_tdata.columns)
    print(f"Gene symbols mapped: {mapped} of {bin_tdata.shape[1]}")

    # Calculate pathway statistics
    pathway_names, frequencies, total_genes, tgenes_count, mean_active, top_genes_list = _calculate_pathway_statistics(pathways_all, pathways, bin_tdata, gene_labels)

    # Sort and prepare for visualization
    sorted_pathways, sorted_freq, sorted_total, sorted_tgenes, sorted_active, sorted_genes = _sort_pathways_by_frequency(pathway_names, frequencies, total_genes, tgenes_count, mean_active, top_genes_list)

    # Create hover text
    hover_text = _create_hover_text(sorted_pathways, sorted_freq, sorted_total, sorted_tgenes, sorted_active, sorted_genes)

    # Create and save plot
    _create_and_save_plot(sorted_pathways, sorted_freq, hover_text, tissue, output_dir)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    analyze_tgenes('GBM', '6. TCGA-GBM')
