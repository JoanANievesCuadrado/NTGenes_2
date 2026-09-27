import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from pathlib import Path
from scipy.stats import gmean
from upsetplot import UpSet, from_indicators

from utils import load_tcga_data, load_pathways, intersect_data_pathway, ge2pe_fast, binarize_data, get_row_genes, get_ensembl_from_gene_name
from tgenes import get_tgenes
from ngenes import get_ngenes


TCGA_FOLDER = Path('../TCGA-DATA')
EXT_FOLDER = Path('data_external')
OUT_FOLDER = Path('figures and tables')

TISSUES = {
    # 'PRAD': '2. TCGA-PRAD',
    'GBM': '6. TCGA-GBM',
    # 'KIRC': '3. TCGA-KIRC_KidneyClearCell',
    # 'LUSC': '4. TCGA-LUSC_LungScamousCell',
    # 'LUAD': '5. TCGA-LUAD_LungAdenoma',
    # 'UCEC': '7. TCGA-UCEC_Uterus',
    # 'PAAD': '8. TCGA-PAAD',
    # 'KIRP': '9. TCGA-KIRP_KidneyPapillary',
    # 'CESC': '10. TCGA-CESC',
    # 'BLCA': '11. TCGA-BLCA_Bladder',
    # 'AdrenalGland': '12. TCGA-AdrenalGland',
    # 'COAD': '13. TCGA-COAD_Colon',
    # 'ESCA': '14. TCGA-ESCA_Esophagus',
    # 'LIHC': '15. TCGA-LIHC_Liver',
    # 'STAD': '16. TCGA-STAD_Stomach',
    # 'THCA': '17. TCGA-THCA_Thyroid',
    # 'BRCA': '18. TCGA-BRCA_Breast',
    # 'HNSC': 'HNSC',
    # 'READ': 'READ',
}


def main():
    pass


if __name__ == '__main__':
    tissue, tissue_folder = 'GBM', '6. TCGA-GBM'
    # tissue, tissue_folder = 'PRAD', '2. TCGA-PRAD'
    OUT_FOLDER = Path('figures and tables') / (tissue + '_heberferon')
    OUT_FOLDER.mkdir(exist_ok=True, parents=True)
    data_path = TCGA_FOLDER / tissue_folder
    data, normal_mask = load_tcga_data(data_path)

    # Get T-genes and N-genes
    (
        tdata, tgenes,
        tgenes_above, tgenes_bellow,
        tgenes_max_above, tgenes_min_bellow,
        tgenes_above_only, tgenes_both, tgenes_bellow_only) = get_tgenes(data, normal_mask)

    (
        ndata, ngenes,
        ngenes_above, ngenes_bellow,
        ngenes_max_above, ngenes_min_bellow,
        ngenes_above_only, ngenes_both, ngenes_bellow_only) = get_ngenes(data, normal_mask)

    # Calculate fold change
    normal = data[normal_mask][tgenes] + 0.1
    tumor = data[~normal_mask][tgenes] + 0.1

    ref = gmean(normal)
    tcga_fc = np.log2(tumor/ref)
    tcga_mean_fc = tcga_fc.mean()

    rg = get_row_genes(Path('../TCGA-DATA/rows_genes2.xlsx'))

    hbrf = pd.read_table(EXT_FOLDER / 'GSE214832' / 'GSE214832.top.table.tsv')
    hbrf_gene_set = set(hbrf['Gene.symbol'].dropna().drop_duplicates().to_list())
    gene_rg_set = set(rg.gene_symbol.dropna().drop_duplicates().to_list())
    hbrf_gene_set = hbrf_gene_set.intersection(gene_rg_set)
    hbrf_logfc = hbrf.dropna(subset=['Gene.symbol']).groupby('Gene.symbol').logFC.mean()
    hbrf_logfc = hbrf_logfc[hbrf_logfc.index.isin(list(hbrf_gene_set))]
    hbrf_logfc.index = get_ensembl_from_gene_name(rg, hbrf_logfc.index)
    hbrf_logfc = hbrf_logfc[hbrf_logfc.index.dropna()]
    hbrf_logfc = hbrf_logfc[np.intersect1d(tgenes, hbrf_logfc.index.to_numpy())]
    idh1_ens = get_ensembl_from_gene_name(rg, ['IDH1'])[0]
    idh1_hbrf_sign = np.sign(hbrf_logfc[idh1_ens])
    idh1_tcga_sign = np.sign(tcga_mean_fc[idh1_ens])
    hbrf_logfc = idh1_hbrf_sign * idh1_tcga_sign * hbrf_logfc

    # --------------------------------------------------------------------------
    a = np.sqrt(tcga_mean_fc[hbrf_logfc.index].pow(2).sum())
    b = np.sqrt(hbrf_logfc.pow(2).sum())
    aa = tcga_mean_fc[hbrf_logfc.index] / a
    bb = hbrf_logfc / b
    print(f'{aa.dot(bb) = }')

    # --------------------------------------------------------------------------

    _, bins = np.histogram(np.concatenate((hbrf_logfc, tcga_mean_fc[hbrf_logfc.index])), bins=30)
    density = False
    tcga_mean_fc[hbrf_logfc.index].hist(bins=bins, label='tcga', alpha=0.5, density=density)
    hbrf_logfc.hist(bins=bins, label='heberferon', alpha=0.5, density=density)
    plt.legend()
    plt.show()

    # --------------------------------------------------------------------------

    bin_tdata = binarize_data(tdata, tgenes_above, tgenes_bellow, tgenes_max_above, tgenes_min_bellow)
    bin_ndata = binarize_data(ndata, ngenes_above, ngenes_bellow, ngenes_max_above, ngenes_min_bellow)

    pathways = load_pathways(EXT_FOLDER / 'pathways/All_pathways.csv')
    pathways_ = pathways.copy()
    bin_tdata, pathways, common_genes = intersect_data_pathway(bin_tdata, pathways, tgenes)
    bin_data_mean = bin_tdata.mean()

    ens_common = np.intersect1d(hbrf_logfc.index, bin_data_mean.index)
    tcga_mean_fc2 = tcga_mean_fc[ens_common].copy()
    tcga_mean_fc2[tcga_mean_fc2.abs() < 1] = 1000

    quotient = hbrf_logfc[ens_common] / tcga_mean_fc2
    bin_data_hbrf = bin_data_mean.copy()
    bin_data_hbrf[quotient[quotient < -0.5].index] = 1
    bin_data_hbrf[quotient[quotient > 0.5].index] = 0

    pw_tcga = ge2pe_fast(bin_data_mean, pathways, mode='proportion')
    pw_hbrf = ge2pe_fast(bin_data_hbrf, pathways, mode='proportion')

    # --------------------------------------------------------------------------

    to_save = {}
    to_save['tcga'] = pw_tcga
    to_save['heberferon'] = pw_hbrf
    to_save['difference'] = pw_tcga - pw_hbrf
    to_save['Pathway size'] = pathways_.groupby('Pathway').count()['Gene']
    to_save['T-Genes'] = pathways.groupby('Pathway').count()['Gene']

    to_one = (quotient < -0.5).astype(int)
    to_one.name = 'to_one'
    to_zero = (quotient > 0.5).astype(int)
    to_zero.name = 'to_zero'

    pathways2 = pathways.copy()
    pathways2.drop_duplicates(inplace=True)

    to_save['to_one'] = (pathways2.set_index('Gene')
                         .merge(to_one, left_index=True, right_index=True)
                         .groupby('Pathway').sum())['to_one']
    to_save['to_zero'] = (pathways2.set_index('Gene')
                          .merge(to_zero, left_index=True, right_index=True)
                          .groupby('Pathway').sum())['to_zero']
    df = pd.DataFrame(to_save)
    df = df.dropna(subset=['tcga'])
    args = np.argsort(-df.difference.abs())
    df = df.iloc[args]
    df.to_csv(OUT_FOLDER / 'pathways_heberferon.csv')

    # --------------------------------------------------------------------------

    args = np.argsort(-pw_tcga.abs())
    plt.plot(pw_tcga[args].to_numpy(), 'o-', color='C2', ms=5, label='TCGA', alpha=0.5)
    plt.plot(pw_hbrf[args].to_numpy(), 'o-', color='C3', ms=5, label='Heberferon', alpha=0.5)
    plt.plot(pw_tcga[args].to_numpy(), 'o-', color='C2', ms=5, alpha=0.5)
    plt.legend()
    plt.xlabel('Pathways')
    plt.ylabel('Fraction of desregulated T-Genes')
    plt.tight_layout()
    plt.savefig(OUT_FOLDER / 'heberferon_dist.pdf')
    plt.show()
