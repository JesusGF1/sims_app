import scanpy as sc
import torch
from scsims import SIMS
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import pandas as pd 
import json 
import numpy as np

st.write("Upload your h5ad")
uploaded_file = st.file_uploader("File upload", type='h5ad')
if uploaded_file is not None:
    with NamedTemporaryFile(dir='.', suffix='.h5ad') as f:
        f.write(uploaded_file.getbuffer())
        st.write(f.name)
        testdata = sc.read_h5ad(f.name)
        if 'testdata' not in st.session_state:
            st.session_state['testdata'] = testdata

    if "testdata" in st.session_state:
        model_checkpoint_folder = "./checkpoint"
        checkpoint_files = [f for f in os.listdir(model_checkpoint_folder) if f.endswith(".ckpt")]
        checkpoint_files = [os.path.join(model_checkpoint_folder, f) for f in checkpoint_files]
        selected_checkpoint = st.selectbox("Select a Model Checkpoint", checkpoint_files, index=None)

        st.write("Selected Checkpoint: ", selected_checkpoint)
        if 'checkpoint' not in st.session_state and selected_checkpoint:
            st.session_state['checkpoint'] = selected_checkpoint

        if "testdata" in st.session_state and "checkpoint" in st.session_state:
            predict = st.button("Predict your cell types")

        if "testdata" in st.session_state and "checkpoint" in st.session_state and predict: 
            selected_checkpoint = st.session_state['checkpoint']
            testdata = st.session_state['testdata']

            if 'run' not in st.session_state:
                st.session_state['run'] = False
            model_run = st.session_state['run']

            if uploaded_file is not None and selected_checkpoint is not None and not model_run:

                print(f"Loading in: {selected_checkpoint}")
                sims = SIMS(weights_path=selected_checkpoint,map_location=torch.device('cpu'))

                st.session_state['model'] = sims

                cell_predictions = sims.predict(testdata, num_workers=0, batch_size=32)
                st.session_state['run'] = True
                st.write("Predictions are done!")
                testdata.obs = testdata.obs.reset_index()
                testdata.obs['cell_predictions'] = cell_predictions["first_pred"]
                testdata.obs['confidence_score'] = cell_predictions["first_prob"]
                st.dataframe(testdata.obs)
                data_as_csv= testdata.obs.to_csv(index=False).encode("utf-8")

            if model_run:
                st.download_button(
                    "Download predictions as CSV", 
                    data_as_csv, 
                    "scSimspredictions.csv",
                    "text/csv",
                )

        visualize = None
        if "testdata" in st.session_state and "checkpoint" in st.session_state and "run" in st.session_state and st.session_state["run"]:
            visualize = st.button("Visualize Predictions")

        if "testdata" in st.session_state and "checkpoint" in st.session_state and visualize:
            testdata = st.session_state['testdata']
            # Add the code to compute UMAP and visualize here
            # Preprocess the data
            # normalize and log 
            sc.pp.normalize_total(testdata, target_sum=1e4)
            sc.pp.log1p(testdata)

            # Perform scaling, PCA, and UMAP
            sc.pp.scale(testdata)
            sc.tl.pca(testdata, n_comps=50)

            sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
            sc.tl.umap(testdata)

            # Visualize the UMAP plot
            st.write("UMAP Visualization with Predictions")
            fig = sc.pl.umap(testdata, color='cell_predictions', palette='tab20', return_fig=True)
            st.pyplot(fig)

        if "testdata" in st.session_state and "checkpoint" in st.session_state:
            explain = st.button("Generate explainability matrix")

        if "testdata" in st.session_state and "checkpoint" in st.session_state and explain:
            selected_checkpoint = st.session_state['checkpoint']
            
            st.write(f"Loading in: {selected_checkpoint}")
            sims = SIMS(weights_path=selected_checkpoint, map_location=torch.device('cpu')) if 'model' not in st.session_state else st.session_state['model']
            explain = sims.explain(testdata, num_workers=0, batch_size=32)[0]
            explain = pd.DataFrame(explain, columns=sims.model.genes)

            # get average gene expression in explainability matrix 
            explain = explain.mean(axis=0)
            # get 10 genes with highest average gene expression
            explain = explain.nlargest(10)
            top10genes = explain.index.tolist()

            # generate the umap plots of the top 10 genes 
            # check if the umap is already calculated in the anndata object 
            if 'X_umap' not in testdata.obsm:
                sc.pp.normalize_total(testdata, target_sum=1e4)
                sc.pp.log1p(testdata)

                # Perform scaling, PCA, and UMAP
                sc.pp.scale(testdata)
                sc.tl.pca(testdata, n_comps=50)

                sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
                sc.tl.umap(testdata)

            # plot a grid of 10 umap plots with the color being the expression of the gene
            st.write("UMAP Visualization with Explainability Matrix")
            fig = sc.pl.umap(testdata, color=top10genes, palette='viridis', ncols=5, return_fig=True)
            st.pyplot(fig)
        # Initialize session state variables
        if "uploaded_json" not in st.session_state:
            st.session_state.uploaded_json = None

        if "gsea_json_data" not in st.session_state:
            st.session_state.gsea_json_data = None

        if "matching_genes" not in st.session_state:
            st.session_state.matching_genes = None

        if "button_sent" not in st.session_state:
            st.session_state.button_sent = False

        # Flag to track if a new file has been uploaded
        new_file_uploaded = False

        # GSEA Pathway Analysis button
        gsea = st.button("GSEA Pathway Visualization")
        if gsea or st.session_state.button_sent:
            if not st.session_state.button_sent:
                st.session_state.button_sent = True

            # Check if a new file has been uploaded
            uploaded_file = st.file_uploader("Upload GSEA JSON", type='json')
            if uploaded_file is not None and uploaded_file != st.session_state.uploaded_json:
                # Reset session state variables related to the uploaded file and analysis
                st.session_state.uploaded_json = uploaded_file
                st.session_state.gsea_json_data = json.load(uploaded_file)
                st.session_state.matching_genes = None
                new_file_uploaded = True

            # Proceed with analysis if a new file is uploaded
            if new_file_uploaded:
                gene_symbols = []
                for key, value in st.session_state.gsea_json_data.items():
                    if 'geneSymbols' in value:
                        gene_symbols.extend(value['geneSymbols'])
                gsea_genes = set(gene_symbols)
                
                # get genes from testdata with nonzero expression
                nonzero_mask = np.asarray(testdata.X.sum(axis=0)).flatten() > 0 
                nonzero_genes = testdata.var.index[nonzero_mask]
                testdata_genes = set(nonzero_genes)

                # find the intersection of genes
                matching_genes = gsea_genes.intersection(testdata_genes)
                if len(matching_genes) == 0:
                    st.write("No matching genes")
                else:
                    st.session_state.matching_genes = matching_genes
                    st.write("Matching Genes:")
                    st.write(matching_genes)
                    
                    # Display "Generating UMAP..." message
                    with st.spinner("Generating UMAP..."):
                        # Check if the UMAP is already calculated in the anndata object 
                        if 'X_umap' not in testdata.obsm:
                            sc.pp.normalize_total(testdata, target_sum=1e4)
                            sc.pp.log1p(testdata)
                            sc.pp.scale(testdata)
                            sc.tl.pca(testdata, n_comps=50)
                            sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
                            sc.tl.umap(testdata)

                    # Plot UMAP with matching genes if they exist
                    if st.session_state.matching_genes:
                        with st.expander("UMAP Visualization with Matching Genes"):
                            fig_placeholder = st.empty()  # Placeholder for the UMAP figure
                            fig = sc.pl.umap(testdata, color=list(st.session_state.matching_genes), palette='tab20', ncols=5, return_fig=True)
                            fig_placeholder.pyplot(fig)
                            
                            # Set the expander state to open regardless of button state
                            st.session_state.umap_expander_open = True