# SIMS App

## Overview
This app was built using Streamlit to facilitate single cell analysis in a user-friendly way. Leveraging our [SIMS](https://github.com/braingeneers/SIMS) model, the app provides features for predicting cell types, generating explainability matrices, and performing gene pathway analysis through just three easy-to-use buttons.

If the app is live, you can access it at [https://sc-sims-app.streamlit.app/](https://sc-sims-app.streamlit.app/).

## Getting Started
To use the app, you only need a single cell dataset in the format of an `.h5ad` file.

## Features

### Predict Cell Types
- **Model Selection**: Choose from a list of pre-trained model checkpoints to predict cell types for your data.
- **Visualization**: Visualize the predicted cell types using UMAP plots.
- **Download**: Export the cell type predictions as a CSV file.

### Generate Explainability Matrix
- **Explainability Calculation**: Calculate the explainability of the selected model.
- **Visualization**: Visualize the explainability matrix with UMAP plots.

### Pathway Analysis using GSEA
- **Data Upload**: Upload a JSON dataset from [GSEA](https://www.gsea-msigdb.org/) containing a known gene pathway for comparison with your dataset.
- **Gene Matching**: Identify and observe which of your genes match genes from known pathways.
- **Visualization**: Visualize these genes and their associations using UMAP plots.

## Dependencies
- [scsims](https://github.com/braingeneers/SIMS)
- [Streamlit](https://streamlit.io/)
