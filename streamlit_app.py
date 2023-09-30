import scanpy as sc
import torch
from scsims import SIMS
import streamlit as st
from tempfile import NamedTemporaryFile
import os

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
    if 'checkpoint' not in st.session_state:
        st.session_state['checkpoint'] = selected_checkpoint
    class_label = st.text_input("Name of the Cell Type column", "CellType") #Class label in this case I think its CellType
    if 'label' not in st.session_state:
        st.session_state['label'] = class_label


if "testdata" in st.session_state and "checkpoint" in st.session_state and "label" in st.session_state: #Delete this if it does not work
    predict = st.button("Predict your cell types")

if "testdata" in st.session_state and "checkpoint" in st.session_state and "label" in st.session_state and predict:
    try:
        selected_checkpoint = st.session_state['checkpoint']
        class_label = st.session_state['label']
        testdata = st.session_state['testdata']
    except:
        st.write("Please select a model and upload your data first")

    if 'run' not in st.session_state:
        st.session_state['run'] = False
    model_run = st.session_state['run']
    if uploaded_file is not None and selected_checkpoint is not None and class_label is not None and model_run is False:
        sims = SIMS(weights_path=selected_checkpoint, class_label=class_label)
        cell_predictions = sims.predict(testdata,num_workers=0)
        st.session_state['run'] = True
        st.write("Predictions are done!")
        testdata.obs = testdata.obs.reset_index()
        testdata.obs['cell_predictions'] = cell_predictions["first_pred"]
        testdata.obs['confidence_score'] = cell_predictions["first_prob"]
        print(testdata.obs)
        st.dataframe(testdata.obs)
        st.download_button('Download predictions', testdata.obs, 'text/csv')
    else:
        st.write("Please select a model and upload your data first")
    

#Casi que funciona, hay que tener cuidado de no repetir los botones ya que cada interaccion reinicia el script

#st.write("""Visualize your data here:""")
#st.dataframe(testdata.obs)
#sc.tl.umap(testdata)
#sc.pl.umap(testdata, color=['cell_predictions'])

#
#streamlit run streamlit_app.py --server.maxUploadSize 1000