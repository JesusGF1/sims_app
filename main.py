import scanpy as sc
import torch
from scsims import SIMS
import streamlit as st
import os

st.write("""Upload your data here:""")
#file = st.file_uploader("upload file", type={"h5ad"})
#if file is not None:
#    testdata = sc.read_h5ad(file)
#else:
#    st.write('Please upload the file which you want to predict')
#st.write(testdata.obs)
data_folder = "./dataset"
data_files = [f for f in os.listdir(data_folder) if f.endswith(".h5ad")]
data_files = [os.path.join(data_folder, f) for f in data_files]
selected_file = st.selectbox("Select the file you want to do label transfer on", data_files)
testdata = sc.read_h5ad(selected_file)


st.write("""Select your model here:""")

model_checkpoint_folder = "./checkpoint"
checkpoint_files = [f for f in os.listdir(model_checkpoint_folder) if f.endswith(".ckpt")]
checkpoint_files = [os.path.join(model_checkpoint_folder, f) for f in checkpoint_files]
selected_checkpoint = st.selectbox("Select a Model Checkpoint", checkpoint_files)
class_label = st.text_input("Name of the Cell Type column", "CellType") #Class label in this case I think its CellType

st.write("""Predict your data here:""")
if st.button('Click here to start the predictions'):
    sims = SIMS(weights_path=selected_checkpoint, class_label=class_label)
    cell_predictions = sims.predict(testdata)
    st.write("Predictions are done!")
    testdata.obs = testdata.obs.reset_index()
    testdata.obs['cell_predictions'] = cell_predictions["first_pred"]
    testdata.obs['confidence_score'] = cell_predictions["first_prob"]
else:
    st.write('Press the button to start the predictions')

st.write("""Visualize your data here:""")
st.dataframe(testdata.obs)
#sc.pl.umap(testdata, color=['cell_predictions'])

# streamlit run main.py --server.maxUploadSize 1000