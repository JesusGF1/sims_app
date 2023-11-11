import scanpy as sc
import torch
from scsims import SIMS
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import time 

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
    selected_checkpoint = st.selectbox("Select a Model Checkpoint", checkpoint_files)
    st.write('You selected:', selected_checkpoint)

    st.session_state['checkpoint'] = selected_checkpoint

if "testdata" in st.session_state and "checkpoint" in st.session_state:
    predict = st.button("Predict your cell types")

if "testdata" in st.session_state and "checkpoint" in st.session_state and predict: 
    try:
        selected_checkpoint = st.session_state['checkpoint']
        testdata = st.session_state['testdata']
    except:
        st.write("Please select a model and upload your data first")

    if 'run' not in st.session_state:
        st.session_state['run'] = False
    model_run = st.session_state['run']
    if uploaded_file is not None and selected_checkpoint is not None and model_run is False:
        print(f"Loading in {selected_checkpoint}")
        sims = SIMS(weights_path=selected_checkpoint,map_location=torch.device('cpu'))
        cell_predictions = sims.predict(testdata, num_workers=0, batch_size=32)
        st.session_state['run'] = True
        st.write("Predictions are done!")
        testdata.obs = testdata.obs.reset_index()
        testdata.obs['cell_predictions'] = cell_predictions["first_pred"]
        testdata.obs['confidence_score'] = cell_predictions["first_prob"]
        st.dataframe(testdata.obs)
        data_as_csv= testdata.obs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV", 
            data_as_csv, 
            "scSimspredictions.csv",
            "text/csv",
        )

visualize = st.button("Visualize Predictions")

if visualize and "testdata" in st.session_state:
    try:
        testdata = st.session_state['testdata']
    except:
        st.write("Please generate predictions first.")
    else:
        # Add the code to compute UMAP and visualize here
        # Preprocess the data
        # normalize and log 
        sc.pp.normalize_total(testdata, target_sum=1e4)
        sc.pp.log1p(testdata)

        sc.pp.highly_variable_genes(testdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        print("Highly variable genes: %d" % sum(testdata.var.highly_variable))

        # Subset for highly variable genes
        testdata = testdata[:, testdata.var['highly_variable']]

        # Perform scaling, PCA, and UMAP
        sc.pp.scale(testdata)
        sc.tl.pca(testdata, n_comps=50)

        sc.pp.neighbors(testdata, n_neighbors=20, n_pcs=30)
        sc.tl.umap(testdata)

        # Visualize the UMAP plot
        st.write("UMAP Visualization with Predictions")
        fig = sc.pl.umap(testdata, color='cell_predictions', palette='tab20', return_fig=True)
        st.pyplot(fig)
