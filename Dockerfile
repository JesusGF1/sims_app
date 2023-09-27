FROM anibali/pytorch:1.13.0-cuda11.8
USER root

WORKDIR /src

RUN sudo apt-get update
RUN sudo apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get --allow-releaseinfo-change update && \
    sudo apt-get install -y --no-install-recommends \
    curl \
    sudo \
    vim

ENV CACHE_BUST=13428901342890

RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


COPY . .