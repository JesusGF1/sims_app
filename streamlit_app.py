import scanpy as sc
import torch
from scsims import SIMS
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import pandas as pd 
import json 
import numpy as np
    
st.write("**:blue[Welcome to SIMS!]**")
uploaded_file = st.file_uploader("Upload your h5ad to get started", type='h5ad')
if uploaded_file is not None:
    with NamedTemporaryFile(dir='.', suffix='.h5ad') as f:
        f.write(uploaded_file.getbuffer())
        testdata = sc.read_h5ad(f.name)
        if 'testdata' not in st.session_state:
            st.session_state['testdata'] = testdata
    
    if "testdata" in st.session_state:
        model_checkpoint_folder = "./checkpoint"
        checkpoint_files = [f for f in os.listdir(model_checkpoint_folder) if f.endswith((".ckpt", ".pt"))]
        display_names = sorted([os.path.splitext(f)[0] for f in checkpoint_files])
        display_selected_checkpoint = st.selectbox("Select a Model Checkpoint", display_names, index=None)
        
        if display_selected_checkpoint:
            if display_selected_checkpoint + ".pt" in checkpoint_files:
                selected_checkpoint = os.path.join(model_checkpoint_folder, display_selected_checkpoint + ".pt")
            elif display_selected_checkpoint + ".ckpt" in checkpoint_files:
                selected_checkpoint = os.path.join(model_checkpoint_folder, display_selected_checkpoint + ".ckpt")
            else:
                selected_checkpoint = None  
        else:
            selected_checkpoint = None  
        
        if selected_checkpoint:
            st.session_state['checkpoint'] = selected_checkpoint
        
    ## Predict Cell Types
        if "testdata" in st.session_state and "checkpoint" in st.session_state:
            predict = st.button("Predict Cell Types")

            # check if predict button is clicked and if model_run is False
            if predict and not st.session_state.get('model_run', False):
                selected_checkpoint = st.session_state['checkpoint']
                testdata = st.session_state['testdata']
                
                loading_text = st.empty()
                loading_text.text(f"Loading in: {selected_checkpoint}")
                
                with st.spinner("Calculating predictions... Hang tight! Processing time varies based on file size."):
                    sims = SIMS(weights_path=selected_checkpoint, map_location=torch.device('cpu'))
                    
                    st.session_state['model'] = sims
                    cell_predictions = sims.predict(testdata, num_workers=0, batch_size=32)
                    st.session_state['run'] = True
                    st.write("Predictions are done!")
                    # testdata.obs = testdata.obs.reset_index()
                    testdata.obs['cell_predictions'] = cell_predictions["first_pred"].values
                    testdata.obs['confidence_score'] = cell_predictions["first_prob"].values
                    st.dataframe(testdata.obs)
                    data_as_csv = testdata.obs.to_csv(index=False).encode("utf-8")

                    st.session_state['model_run'] = True    # set model_run to True in session state
                    st.session_state['data_as_csv'] = data_as_csv   # set data_as_csv in session state
                    
                    loading_text.empty()

        if "model_run" in st.session_state and st.session_state['model_run']:
            data_as_csv = st.session_state.get('data_as_csv')

            if data_as_csv is not None:
                st.download_button(
                    "Download Predictions as CSV", 
                    data_as_csv, 
                    "scSimspredictions.csv",
                    "text/csv",
                )
                st.session_state['model_run'] = False   # reset model_run to false so prediction process can 
                                                        # be triggered again after it has been completed
                
    ## Visualize Predictions
        visualize = None
        if "testdata" in st.session_state and "checkpoint" in st.session_state and "run" in st.session_state and st.session_state["run"]:
            visualize = st.button("Visualize Predictions")

        if "testdata" in st.session_state and "checkpoint" in st.session_state and visualize:
            testdata = st.session_state['testdata']
            
            with st.spinner("Creating UMAP..."):
                if 'X_umap' not in testdata.obsm:
                    sc.pp.normalize_total(testdata, target_sum=1e4)
                    sc.pp.log1p(testdata)

                    sc.pp.scale(testdata)
                    sc.tl.pca(testdata, n_comps=50)

                    sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
                    sc.tl.umap(testdata)

            st.write("UMAP Visualization with Predictions")
            fig = sc.pl.umap(testdata, color='cell_predictions', palette='tab20', return_fig=True)
            st.pyplot(fig)
             
    ## Explainability Matrix
        if "testdata" in st.session_state and "checkpoint" in st.session_state:
            explain = st.button("Generate Explainability Matrix")

        if "testdata" in st.session_state and "checkpoint" in st.session_state and explain:
            selected_checkpoint = st.session_state['checkpoint']
    
            loading_text = st.empty()
            loading_text.text(f"Loading in: {selected_checkpoint}")
            
            with st.spinner("Generating matrix... Hang tight! Processing time varies based on file size."):
                sims = SIMS(weights_path=selected_checkpoint, map_location=torch.device('cpu')) if 'model' not in st.session_state else st.session_state['model']
                explain = sims.explain(testdata, num_workers=0, batch_size=32)[0]
                explain = pd.DataFrame(explain, columns=sims.model.genes)

                # get average gene expression in explainability matrix 
                explain = explain.mean(axis=0)
                # get 10 genes with highest average gene expression
                explain = explain.nlargest(10)
                top10genes = explain.index.tolist()
                
                loading_text.empty()

            # generate the umap plots of the top 10 genes 
            # check if the umap is already calculated in the anndata object 
            with st.spinner("Almost there! Creating UMAP..."):
                if 'X_umap' not in testdata.obsm:
                    sc.pp.normalize_total(testdata, target_sum=1e4)
                    sc.pp.log1p(testdata)

                    # perform scaling, PCA, and UMAP
                    sc.pp.scale(testdata)
                    sc.tl.pca(testdata, n_comps=50)

                    sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
                    sc.tl.umap(testdata)

            # plot a grid of 10 umap plots with the color being the expression of the gene
            st.write("UMAP Visualization with Explainability Matrix")
            fig = sc.pl.umap(testdata, color=top10genes, palette='viridis', ncols=5, return_fig=True)
            st.pyplot(fig)

    ## GSEA Pathway Analysis
        if "uploaded_json" not in st.session_state:
            st.session_state.uploaded_json = None

        if "gsea_json_data" not in st.session_state:
            st.session_state.gsea_json_data = None

        if "matching_genes" not in st.session_state:
            st.session_state.matching_genes = None

        if "button_sent" not in st.session_state:
            st.session_state.button_sent = False

        new_file_uploaded = False   # flag to track if a new file has been uploaded

        # GSEA Pathway Analysis button
        if "testdata" in st.session_state and "checkpoint" in st.session_state:
            gsea = st.button("GSEA Pathway Visualization")
            if gsea or st.session_state.button_sent:
                if not st.session_state.button_sent:
                    st.session_state.button_sent = True
    
                uploaded_file = st.file_uploader("Upload a JSON dataset of gene pathways from [GSEA](https://www.gsea-msigdb.org/gsea/downloads.jsp) to compare with your genes and visualize with UMAP")
                if uploaded_file is not None and uploaded_file != st.session_state.uploaded_json:
                    # Reset session state variables related to the uploaded file and analysis
                    st.session_state.uploaded_json = uploaded_file
                    st.session_state.gsea_json_data = json.load(uploaded_file)
                    st.session_state.matching_genes = None
                    new_file_uploaded = True

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
                        
                        with st.spinner("Creating UMAP..."):
                            # check if the UMAP is already calculated in the anndata object 
                            if 'X_umap' not in testdata.obsm:
                                sc.pp.normalize_total(testdata, target_sum=1e4)
                                sc.pp.log1p(testdata)
                                sc.pp.scale(testdata)
                                sc.tl.pca(testdata, n_comps=50)
                                sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
                                sc.tl.umap(testdata)

                        # plot UMAP with matching genes if they exist
                        if st.session_state.matching_genes:
                            with st.expander("UMAP Visualization with Matching Genes"):
                                fig_placeholder = st.empty()  # Placeholder for the UMAP figure
                                fig = sc.pl.umap(testdata, color=list(st.session_state.matching_genes), palette='tab20', ncols=5, return_fig=True)
                                fig_placeholder.pyplot(fig)
                                