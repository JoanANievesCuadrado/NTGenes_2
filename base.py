import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from pathlib import Path
from scipy.stats import gmean
from typing import List, Optional
from upsetplot import UpSet, from_indicators


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


def load_tcga_data(data_path: str | Path) -> pd.DataFrame:
    sample_path = data_path / 'sample.xls'
    data_path = data_path / 'data'

    sample = pd.read_excel(sample_path)
    normal_mask = sample['Sample Type'] == 'Solid Tissue Normal'

    # Loading one data file
    def get_data(row: pd.Series) -> pd.Series:
        filename = row['File Name']
        file = pd.read_table(data_path / filename, names=['gene_id', 'value'])
        return file['value']

    data = sample.apply(get_data, axis=1)
    
    filename_0 = sample['File Name'][0]
    file_0 = pd.read_table(data_path / filename_0, names=['gene_id', 'value'])
    gene_id = file_0.gene_id
    gene_id = np.vectorize(lambda x: x.split('.')[0])(gene_id)
    sample_id = sample['Sample ID']

    data.index = sample_id
    data.columns = gene_id
    normal_mask.index = sample_id

    return data, normal_mask


def get_ngenes(data, normal_mask):
    normal_data = data[normal_mask]
    tumor_data = data[~normal_mask]
    tumor_min, tumor_max = tumor_data.min(), tumor_data.max()
    n_normal = normal_data.shape[0]

    f_bellow = (normal_data < tumor_min - 0.1).sum() / n_normal
    f_above = (normal_data > tumor_max + 0.1).sum() / n_normal

    ibellow, = np.where(f_bellow > 0.05)
    iabove, = np.where(f_above > 0.05)

    genes_above_set = set(data.columns[iabove])
    genes_bellow_set = set(data.columns[ibellow])

    # Classify genes according to Mathematica logic
    genes_both = np.array(list(genes_above_set & genes_bellow_set))  # "no"
    genes_above_only = np.array(list(genes_above_set - genes_both))  # "na"
    genes_bellow_only = np.array(list(genes_bellow_set - genes_both))  # "nb"

    genes_above, max_above = data.columns[iabove], tumor_max.iloc[iabove]
    genes_bellow, min_bellow = data.columns[ibellow], tumor_min.iloc[ibellow]
    ngenes = np.array(list(genes_above_set | genes_bellow_set))
    ndata = normal_data[ngenes]

    return ndata, ngenes, genes_above, genes_bellow, max_above, min_bellow, genes_above_only, genes_both, genes_bellow_only


def get_tgenes(data, normal_mask):
    normal_data = data[normal_mask]
    tumor_data = data[~normal_mask]
    normal_min, normal_max = normal_data.min(), normal_data.max()
    n_tumor = tumor_data.shape[0]

    f_bellow = (tumor_data < normal_min - 0.1).sum() / n_tumor
    f_above = (tumor_data > normal_max + 0.1).sum() / n_tumor

    ibellow, = np.where(f_bellow > 0.1)
    iabove, = np.where(f_above > 0.1)

    genes_above_set = set(data.columns[iabove])
    genes_bellow_set = set(data.columns[ibellow])

    # Classify genes according to Mathematica logic
    genes_both = np.array(list(genes_above_set & genes_bellow_set))  # "to"
    genes_above_only = np.array(list(genes_above_set - genes_both))  # "ta"
    genes_bellow_only = np.array(list(genes_bellow_set - genes_both))  # "tb"

    genes_above, max_above = data.columns[iabove], normal_max.iloc[iabove]
    genes_bellow, min_bellow = data.columns[ibellow], normal_min.iloc[ibellow]
    tgenes = np.array(list(genes_above_set | genes_bellow_set))
    tdata = tumor_data[tgenes]

    return tdata, tgenes, genes_above, genes_bellow, max_above, min_bellow, genes_above_only, genes_both, genes_bellow_only


def binarize_data(tdata, genes_above, genes_bellow, max_above, min_bellow):
    bin_data = pd.DataFrame(0, columns=tdata.columns, index=tdata.index)
    bin_data.loc[:, genes_above] = (tdata[genes_above] > max_above).astype(int)
    bin_data.loc[:, genes_bellow] = (tdata[genes_bellow] < min_bellow).astype(int)
    
    return bin_data


def load_pathways():
    path = EXT_FOLDER / Path('pathways/All_pathways.csv')
    pathways = pd.read_csv(path, names=['Gene', 'Pathway'])
    return pathways


def intersect_data_pathway(bin_data, pathways, tgenes):
    pw_genes = pathways.Gene.unique()
    common_genes = np.intersect1d(pw_genes, tgenes)
    bin_data = bin_data[common_genes]
    pathways = pathways[pathways.Gene.isin(common_genes)]

    return bin_data, pathways, common_genes


def ge2pe_fast(df_expr, pathways, mode="proportion"):
    pathways_unique = pathways.drop_duplicates(subset=['Gene', 'Pathway'])
    
    pathway_matrix = pd.crosstab(pathways_unique['Gene'], pathways_unique['Pathway'])
    
    df_pathway = df_expr.dot(pathway_matrix)
    
    if mode == "presence":
        df_pathway = (df_pathway > 0).astype(int)
    elif mode == "proportion":
        pw_count = pathway_matrix.sum(axis=0)
        df_pathway = df_pathway.divide(pw_count)
    
    return df_pathway


def get_row_genes():
    row_genes_path = Path('../TCGA-DATA/rows_genes2.xlsx')
    rg = pd.read_excel(row_genes_path)
    return rg


def get_gene_name_from_ensembl(rg: pd.DataFrame, ensembl_list: List | np.ndarray):
    return rg.set_index('ensembl').loc[ensembl_list].gene_symbol.to_numpy()


def get_ensembl_from_gene_name(rg: pd.DataFrame, gene_list: List | np.ndarray):
    return rg.set_index('gene_symbol').loc[gene_list].ensembl.to_numpy()


def main():
    pass


if __name__ == '__main__':
#     main()
# else:
    tissue, tissue_folder = 'GBM', '6. TCGA-GBM'
    # tissue, tissue_folder = 'PRAD', '2. TCGA-PRAD'
    OUT_FOLDER = Path('figures and tables') / (tissue + '_heberferon')
    OUT_FOLDER.mkdir(exist_ok=True, parents=True)
    data_path = TCGA_FOLDER / tissue_folder
    data, normal_mask = load_tcga_data(data_path)
    (
        tdata, tgenes,
        genes_above, genes_bellow,
        max_above, min_bellow,
        genes_above_only, genes_both, genes_bellow_only) = get_tgenes(data, normal_mask)

    normal = data[normal_mask][tgenes] + 0.1
    tumor = data[~normal_mask][tgenes] + 0.1

    ref = gmean(normal)
    tcga_fc = np.log2(tumor/ref)
    tcga_mean_fc = tcga_fc.mean()

    rg = get_row_genes()

    hbrf = pd.read_table(EXT_FOLDER / 'GSE214832' / 'GSE214832.top.table.tsv')
    hbrf_gene_set = set(hbrf['Gene.symbol'].dropna().drop_duplicates().to_list())
    gene_rg_set = set(rg.gene_symbol.dropna().drop_duplicates().to_list())
    hbrf_gene_set = hbrf_gene_set.intersection(gene_rg_set)
    # hbrf_ens_list = get_ensembl_from_gene_name(rg, list(hbrf_gene_set))
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

    bin_data = binarize_data(tdata, genes_above, genes_bellow, max_above, min_bellow)
    pathways = load_pathways()
    pathways_ = pathways.copy()
    bin_data, pathways, common_genes = intersect_data_pathway(bin_data, pathways, tgenes)
    bin_data_mean = bin_data.mean()

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
    # df = df.sort_values(by='difference', ascending=False)
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
