import scanpy as sc
import torch
from scsims import SIMS
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import pandas as pd
import json
import numpy as np


def compute_umap(adata):
    """Run normalize -> log1p -> scale -> PCA -> neighbors -> UMAP on `adata`,
    in place, picking parameter values that are safe for the actual dataset
    shape. Hardcoded n_comps=50 / n_pcs=30 / n_neighbors=20 used to crash on
    any dataset with <=50 cells (e.g. the small test fixture)."""
    if "X_umap" in adata.obsm:
        return  # already computed

    n_cells, n_genes = adata.shape
    if n_cells < 4:
        raise ValueError(
            f"Need at least 4 cells to compute a UMAP, got {n_cells}."
        )

    # PCA n_components must be strictly less than min(n_samples, n_features).
    n_comps = min(50, min(n_cells, n_genes) - 1)
    n_pcs = min(30, n_comps)
    n_neighbors = min(20, n_cells - 1)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)


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
        st.session_state['checkpoint'] = selected_checkpoint

    # Action selection
    if st.session_state['checkpoint']:
        action = st.selectbox("Choose an action", ["Select Action", "Predict Cell Types", "Generate Explainability Matrix", "Gene Pathway Visualization"])
        if action == "Predict Cell Types":
            if "testdata" in st.session_state:
                if not st.session_state.get('model_run', False):
                    selected_checkpoint = st.session_state['checkpoint']
                    testdata = st.session_state['testdata']

                    loading_text = st.empty()
                    loading_text.text(f"Loading in: {selected_checkpoint}")

                    try:
                        with st.spinner(":blue[Calculating predictions... Hang tight! Processing time varies based on file size.]"):
                            # no_explain=True skips building the (post_embed_dim x
                            # input_dim) reducing matrix that pytorch_tabnet
                            # creates by default. For the shipped checkpoints
                            # (input_dim ~= 34k) this saves ~1 GB of RAM at load
                            # time, which is the difference between fitting in
                            # the Streamlit Community Cloud free tier (1 GB) and
                            # being SIGKILLed mid-load with no Python traceback.
                            sims = SIMS(
                                weights_path=selected_checkpoint,
                                map_location=torch.device('cpu'),
                                no_explain=True,
                            )

                            st.session_state['model'] = sims
                            cell_predictions = sims.predict(testdata, num_workers=0, batch_size=32)
                            st.session_state['run'] = True
                            st.caption("Predictions are done!")
                            # scsims.SIMS.predict() returns top-k predictions as
                            # `pred_0..pred_{k-1}` and `prob_0..prob_{k-1}`.
                            testdata.obs['cell_predictions'] = cell_predictions["pred_0"].values
                            testdata.obs['confidence_score'] = cell_predictions["prob_0"].values
                            st.dataframe(testdata.obs)
                            data_as_csv = testdata.obs.to_csv(index=False).encode("utf-8")

                            st.session_state['model_run'] = True
                            st.session_state['data_as_csv'] = data_as_csv

                            loading_text.empty()
                    except Exception as e:
                        # Surface any failure (OOM kill from the OS won't reach
                        # this handler, but every other failure mode will).
                        # Without this, exceptions get swallowed by Streamlit's
                        # script-runner and the user just sees "Oh no" with no
                        # actionable details.
                        loading_text.empty()
                        st.error(
                            f"Prediction failed: **{type(e).__name__}**: {e}\n\n"
                            "If you're on Streamlit Community Cloud's free tier, "
                            "this is most likely an out-of-memory kill on a large "
                            "checkpoint. The shipped MGE_cortex / Allen / Velasco "
                            "checkpoints have ~34k input genes and need ~1.3-2.6 GB "
                            "of RAM, which exceeds the 1 GB free tier limit. "
                            "Try a smaller checkpoint (`human.pt`, `chimp.pt`, etc.) "
                            "or run the app locally."
                        )
                        st.exception(e)

                if "model_run" in st.session_state and st.session_state['model_run']:
                    # Show buttons for visualizing and downloading predictions
                    visualize = st.button("Visualize Predictions")
                    if visualize:
                        testdata = st.session_state['testdata']
                        
                        try:
                            with st.spinner(":blue[Creating UMAP...]"):
                                compute_umap(testdata)
                        except ValueError as e:
                            st.error(f"Could not compute UMAP: {e}")
                        else:
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
                selected_checkpoint = st.session_state['checkpoint']

                st.warning(
                    ":warning: **Heads up:** building the interpretability "
                    "matrix needs a lot more memory than the predict step "
                    "because pytorch_tabnet has to materialize a dense "
                    "(post_embed_dim x input_dim) reducing matrix. For the "
                    "shipped checkpoints (~34k input genes) this can push "
                    "RAM use over the 1 GB limit on the Streamlit Community "
                    "Cloud free tier and crash the worker. If it fails, "
                    "try a smaller checkpoint or run the app locally."
                )

                loading_text = st.empty()
                loading_text.text(f"Loading in: {selected_checkpoint}")

                try:
                    with st.spinner(":blue[Generating matrix... Hang tight! Processing time varies based on file size.]"):
                        # Need a model with the explain reducing matrix built.
                        # The cached `model` from the Predict path was loaded
                        # with no_explain=True (memory optimization), so we
                        # can't reuse it here -- have to reload.
                        cached = st.session_state.get('model')
                        if cached is not None and hasattr(cached.model, 'reducing_matrix'):
                            sims = cached
                        else:
                            sims = SIMS(weights_path=selected_checkpoint, map_location=torch.device('cpu'))
                            st.session_state['model'] = sims

                        explain = sims.explain(testdata, num_workers=0, batch_size=32)[0]
                        explain = pd.DataFrame(explain, columns=sims.model.genes)

                        # get average gene expression in explainability matrix
                        explain = explain.mean(axis=0)
                        # get 10 genes with highest average gene expression
                        explain = explain.nlargest(10)
                        top10genes = explain.index.tolist()

                        loading_text.empty()
                except Exception as e:
                    loading_text.empty()
                    st.error(
                        f"Explainability matrix failed: **{type(e).__name__}**: {e}\n\n"
                        "The explainability path needs even more memory than the "
                        "predict path because pytorch_tabnet builds a dense "
                        "(post_embed_dim x input_dim) reducing matrix. On the "
                        "Streamlit Community Cloud free tier (1 GB) the larger "
                        "shipped checkpoints will OOM during this step."
                    )
                    st.exception(e)
                    st.stop()

                st.write("Top genes selected for explainability matrix:", top10genes)

                # generate the UMAP plot with a try-except block to handle missing genes
                try:
                    with st.spinner(":blue[Creating UMAP...]"):
                        compute_umap(testdata)

                    st.write("UMAP Visualization with Explainability Matrix")
                    fig = sc.pl.umap(testdata, color=top10genes, palette='viridis', ncols=5, return_fig=True, use_raw=False)
                    st.pyplot(fig)

                except ValueError as e:
                    st.error(f"Could not compute UMAP: {e}")
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
                    
                    try:
                        with st.spinner(":blue[Creating UMAP visualization...]"):
                            compute_umap(testdata)
                    except ValueError as e:
                        st.error(f"Could not compute UMAP: {e}")
                    else:
                        # Plot UMAP with matching genes if they exist
                        if st.session_state.matching_genes:
                            with st.expander("UMAP Visualization with Matching Genes"):
                                fig_placeholder = st.empty()  # Placeholder for the UMAP figure
                                fig = sc.pl.umap(testdata, color=list(st.session_state.matching_genes), palette='tab20', ncols=5, return_fig=True)
                                fig_placeholder.pyplot(fig)
                            