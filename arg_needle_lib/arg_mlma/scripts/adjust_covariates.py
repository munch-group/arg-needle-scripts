import click
import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

@click.command(help="Saves rank-normalized residuals of input phenotypes given covariates.")
@click.option("--phenotypes", required=True, help="File in PLINK phenotype format containing phenotypes")
@click.option("--covariates", required=True, help="File in PLINK phenotype format containing covariates")
@click.option("--out", required=True, help="Output file")
def adjust(phenotypes, covariates, out):
    # Read input files
    pheno_df = pd.read_table(phenotypes, delim_whitespace=True)
    covar_df = pd.read_table(covariates, delim_whitespace=True)

    # Make sure input files are OK
    if "IID" not in pheno_df.columns:
        raise RuntimeError(f"No IID column found in {phenotypes}")
    if "IID" not in covar_df.columns:
        raise RuntimeError(f"No IID column found in {covariates}")
    if not (covar_df.IID == pheno_df.IID).all():
        raise RuntimeError(f"IID columns in {phenotypes} and {covariates} don't match")
    if pheno_df.shape[0] <= 1:
        raise RuntimeError(f"No phenotypes found in {phenotypes}")
    if covar_df.shape[0] <= 1:
        raise RuntimeError(f"No covariates found in {covariates}")
    logging.info(f"Found phenotypes {', '.join([p for p in pheno_df.columns if p != 'IID'])}")
    logging.info(f"Found the following covariates:")
    for p in pheno_df.columns:
        if p != "IID" and not pd.api.types.is_numeric_dtype(pheno_df[p].dtype):
            raise RuntimeError(f"Phenotype {p} is not numeric")

    # Make one-hot encodings for categorical covariates
    new_covar_df = covar_df.copy()
    for c in covar_df.columns:
        if c == "IID":
            continue
        numeric = pd.api.types.is_numeric_dtype(covar_df[c].dtype)
        if numeric:
            logging.info(f"\t{c} (numeric)")
        else:
            logging.info(f"\t{c} (non-numeric)")
            dummies = pd.get_dummies(covar_df[c], prefix=c)
            new_covar_df = new_covar_df.join(dummies)
            new_covar_df = new_covar_df.drop(c, axis=1)
    del covar_df

    # Find NaNs in the covariates
    full_covar_matrix = new_covar_df.drop("IID", axis=1).to_numpy()
    pheno_matrix = pheno_df.drop("IID", axis=1).to_numpy()
    covar_nanfilter = ~np.isnan(full_covar_matrix).max(axis=1) # | np.isnan(pheno_matrix).max(axis=1))
    logging.info(f"Omitting {len(covar_nanfilter) - covar_nanfilter.sum()} samples due missing covariates")

    # Initialize output
    out_df = pd.DataFrame({"IID": pheno_df.IID})
    out_vals = np.empty(pheno_matrix.shape)
    out_vals[:] = np.nan

    # Iterate over phenotypes
    for i, p in enumerate(pheno_df.drop("IID", axis=1).columns):
        # Find non-missing values for current phenotype
        pheno_nanfilter = ~np.isnan(pheno_df[p].to_numpy())
        nanfilter = covar_nanfilter * pheno_nanfilter
        pheno_vector = pheno_df[p].to_numpy()[nanfilter]
        covar_matrix = full_covar_matrix[nanfilter]

        # Linear regression
        inverse_matrix = np.linalg.inv(np.dot(covar_matrix.transpose(), covar_matrix))
        prediction = np.dot(covar_matrix.transpose(), pheno_vector)
        prediction = np.dot(inverse_matrix, prediction)
        prediction = np.dot(covar_matrix, prediction)
        residuals = pheno_vector - prediction

        # Rank-normalize residuals
        ranks = residuals.argsort().argsort()
        quantiles = (ranks + 0.5) / len(ranks)
        transformed = norm.ppf(quantiles)
        out_vals[nanfilter, i] = transformed

    # Save normalized residuals
    for i, p in enumerate(pheno_df.drop("IID", axis=1).columns):
        out_df[p] = out_vals[:, i]
    logging.info(f"Writing normalized residuals to {out}")
    out_df.to_csv(out, sep=" ", index=False, na_rep='NaN')

if __name__ == "__main__":
    adjust()
