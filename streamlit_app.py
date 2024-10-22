import scanpy as sc
import torch
from scsims import SIMS
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import pandas as pd 
import json 
import numpy as np

st.subheader("**:red[Welcome to SIMS!]**")

# Upload h5ad file
uploaded_file = st.file_uploader("Upload your h5ad to get started", type='h5ad')

if uploaded_file is not None:
    with NamedTemporaryFile(dir='.', suffix='.h5ad') as f:
        f.write(uploaded_file.getbuffer())
        testdata = sc.read_h5ad(f.name)
        st.session_state['testdata'] = testdata

    model_checkpoint_folder = "./checkpoint"
    checkpoint_files = [f for f in os.listdir(model_checkpoint_folder) if f.endswith((".ckpt", ".pt"))]
    display_names = sorted([os.path.splitext(f)[0] for f in checkpoint_files])
    
    # Checkpoint selection
    if 'checkpoint' not in st.session_state:
        st.session_state['checkpoint'] = None

    # Create a select box for model checkpoints
    checkpoint = st.selectbox(
        "Select a Model Checkpoint",
        ["Select a checkpoint"] + display_names,
        index=0 if st.session_state['checkpoint'] is None else display_names.index(
            os.path.splitext(os.path.basename(st.session_state['checkpoint']))[0]
        ) if os.path.basename(st.session_state['checkpoint']) in display_names else 0
    )

    if checkpoint and st.session_state['checkpoint'] != checkpoint:
        if checkpoint + ".pt" in checkpoint_files:
            selected_checkpoint = os.path.join(model_checkpoint_folder, checkpoint + ".pt")
        elif checkpoint + ".ckpt" in checkpoint_files:
            selected_checkpoint = os.path.join(model_checkpoint_folder, checkpoint + ".ckpt")
        else:
            selected_checkpoint = None
        # st.session_state['checkpoint'] = selected_checkpoint

    # Action selection
    if st.session_state['checkpoint']:
        action = st.selectbox("Choose an action", ["Select Action", "Predict Cell Types", "Generate Explainability Matrix", "Gene Pathway Visualization"])
        if action == "Predict Cell Types":
            if "testdata" in st.session_state:
                if not st.session_state.get('model_run', False):
                    # selected_checkpoint = st.session_state['checkpoint']
                    testdata = st.session_state['testdata']
                    
                    loading_text = st.empty()
                    loading_text.text(f"Loading in: {selected_checkpoint}")
                    
                    with st.spinner(":blue[Calculating predictions... Hang tight! Processing time varies based on file size.]"):
                        sims = SIMS(weights_path=selected_checkpoint, map_location=torch.device('cpu'))
                        
                        st.session_state['model'] = sims
                        cell_predictions = sims.predict(testdata, num_workers=0, batch_size=32)
                        st.session_state['run'] = True
                        st.caption("Predictions are done!")
                        testdata.obs['cell_predictions'] = cell_predictions["first_pred"].values
                        testdata.obs['confidence_score'] = cell_predictions["first_prob"].values
                        st.dataframe(testdata.obs)
                        data_as_csv = testdata.obs.to_csv(index=False).encode("utf-8")

                        st.session_state['model_run'] = True
                        st.session_state['data_as_csv'] = data_as_csv
                        
                        loading_text.empty()

                if "model_run" in st.session_state and st.session_state['model_run']:
                    # Show buttons for visualizing and downloading predictions
                    visualize = st.button("Visualize Predictions")
                    if visualize:
                        testdata = st.session_state['testdata']
                        
                        with st.spinner(":blue[Creating UMAP...]"):
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
                    
                    if st.session_state.get('data_as_csv'):
                        st.download_button(
                            "Download Predictions as CSV", 
                            st.session_state['data_as_csv'], 
                            "scSimspredictions.csv",
                            "text/csv",
                        )
                    st.session_state['model_run'] = False

        elif action == "Generate Explainability Matrix":
            if "testdata" in st.session_state and st.session_state['checkpoint']:
                # selected_checkpoint = st.session_state['checkpoint']

                loading_text = st.empty()
                loading_text.text(f"Loading in: {selected_checkpoint}")

                with st.spinner(":blue[Generating matrix... Hang tight! Processing time varies based on file size.]"):
                    sims = SIMS(weights_path=selected_checkpoint, map_location=torch.device('cpu')) if 'model' not in st.session_state else st.session_state['model']
                    explain = sims.explain(testdata, num_workers=0, batch_size=32)[0]
                    explain = pd.DataFrame(explain, columns=sims.model.genes)

                    # get average gene expression in explainability matrix 
                    explain = explain.mean(axis=0)
                    # get 10 genes with highest average gene expression
                    explain = explain.nlargest(10)
                    top10genes = explain.index.tolist()

                    loading_text.empty()

                st.write("Top genes selected for explainability matrix:", top10genes)

                # generate the UMAP plot with a try-except block to handle missing genes
                try:
                    with st.spinner(":blue[Creating UMAP...]"):
                        if 'X_umap' not in testdata.obsm:
                            sc.pp.normalize_total(testdata, target_sum=1e4)
                            sc.pp.log1p(testdata)

                            # perform scaling, PCA, and UMAP
                            sc.pp.scale(testdata)
                            sc.tl.pca(testdata, n_comps=50)

                            sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
                            sc.tl.umap(testdata)

                    st.write("UMAP Visualization with Explainability Matrix")
                    fig = sc.pl.umap(testdata, color=top10genes, palette='viridis', ncols=5, return_fig=True, use_raw=False)
                    st.pyplot(fig)
                    
                except KeyError as e:
                    # extract all gene identifiers causing errors
                    missing_genes = []
                    genes_to_check = top10genes  

                    for gene in genes_to_check:
                        if gene not in testdata.var_names:
                            missing_genes.append(gene)
                    
                    if missing_genes:   # display the list of genes causing errors
                        st.error(f"The following genes are missing in the dataset and will be excluded: {', '.join(missing_genes)}")
                    
                    # proceed with visualizing available genes if any
                    top10genes_filtered = [gene for gene in top10genes if gene in testdata.var_names]
                    if top10genes_filtered:
                        with st.spinner(":blue[Creating UMAP with available genes...]"):
                            st.write("UMAP Visualization with Explainability Matrix (Filtered)")
                            fig = sc.pl.umap(testdata, color=top10genes_filtered, palette='viridis', ncols=5, return_fig=True)
                            st.pyplot(fig)
                    else:
                        st.error("No valid genes found for visualization. Please check your datasets.")

        elif action == "Gene Pathway Visualization":
            if "input_method" not in st.session_state:
                st.session_state.input_method = None

            if "uploaded_json" not in st.session_state:
                st.session_state.uploaded_json = None

            if "gsea_json_data" not in st.session_state:
                st.session_state.gsea_json_data = None

            if "matching_genes" not in st.session_state:
                st.session_state.matching_genes = None

            if "button_sent" not in st.session_state:
                st.session_state.button_sent = False

            # reset `button_sent` to `False` when the action is selected
            st.session_state.button_sent = False

            st.caption("Compare genes from your expression data with those from known pathways. You can upload a dataset of gene pathways from [GSEA](https://www.gsea-msigdb.org/gsea/downloads.jsp) in JSON format, or manually enter your own list of genes.")

            # select box for choosing input method
            st.session_state.input_method = st.selectbox(
                "Choose a method",
                options=["Select method", "Upload JSON File", "Enter Genes Manually"]
            )

            # show widgets based on the selected input method
            if st.session_state.input_method == "Upload JSON File":
                uploaded_file = st.file_uploader(
                    "Upload a JSON dataset of gene pathways from [GSEA](https://www.gsea-msigdb.org/gsea/downloads.jsp) to compare with your genes and visualize with UMAP",
                    type='json'
                )

                if uploaded_file is not None and uploaded_file != st.session_state.uploaded_json:
                    st.session_state.uploaded_json = uploaded_file
                    st.session_state.gsea_json_data = json.load(uploaded_file)
                    st.session_state.matching_genes = None
                    st.session_state.button_sent = True

            elif st.session_state.input_method == "Enter Genes Manually":
                gene_input = st.text_input("Enter a list of genes separated by commas:")

                if gene_input:
                    gene_list = [gene.strip() for gene in gene_input.split(',')]
                    st.caption(":orange[You entered the following genes:]")
                    st.write(gene_list)
                    st.session_state.gsea_json_data = {"manual_genes": gene_list}
                    st.session_state.matching_genes = None
                    st.session_state.button_sent = True

            # process the input and generate visualization
            if st.session_state.button_sent:
                if st.session_state.input_method == "Upload JSON File":
                    # extract genes from uploaded JSON
                    gene_symbols = []
                    for key, value in st.session_state.gsea_json_data.items():
                        if 'geneSymbols' in value:
                            gene_symbols.extend(value['geneSymbols'])
                    gsea_genes = set(gene_symbols)
                
                elif st.session_state.input_method == "Enter Genes Manually":
                    gsea_genes = set(st.session_state.gsea_json_data.get("manual_genes", []))
                
                nonzero_mask = np.asarray(testdata.X.sum(axis=0)).flatten() > 0 
                nonzero_genes = testdata.var.index[nonzero_mask]
                testdata_genes = set(nonzero_genes)

                # find the intersection of genes
                matching_genes = gsea_genes.intersection(testdata_genes)
                if len(matching_genes) == 0:
                    st.caption(":red[No matching genes found in dataset.]")
                else:
                    st.session_state.matching_genes = matching_genes
                    st.caption(":green[Genes found in dataset:]")
                    st.write(matching_genes)
                    
                    with st.spinner(":blue[Creating UMAP visualization...]"):
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