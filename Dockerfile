FROM anibali/pytorch:1.10.2-cuda11.3
USER root

WORKDIR ./app

RUN sudo apt-get update
RUN sudo apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get --allow-releaseinfo-change update && \
    sudo apt-get install -y --no-install-recommends \
    curl \
    sudo \
    vim

RUN curl -L https://bit.ly/glances | /bin/bash

RUN pip install matplotlib \
    seaborn \
    pytorch-lightning \
    comet_ml \
    wandb \
    pytorch-tabnet \
    scanpy \
    anndata \
    sklearn \
    boto3 \ 
    tenacity \ 
    pandas \
    plotly \
    scipy

RUN pip install git+https://github.com/braingeneers/sims

#RUN pip install streamlit==1.27.0 #It does not work :(
RUN pip install streamlit==1.12.0 
RUN pip install altair==4.0 

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


COPY . .