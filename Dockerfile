FROM continuumio/miniconda3

WORKDIR /app

# Salin semua file proyek
COPY . /app

# Instal dependencies
RUN conda install -c conda-forge --name base python=3.10 pip && \
    pip install mlflow dagshub

# Set environment variables DagsHub sebagai default
ENV MLFLOW_TRACKING_URI=https://dagshub.com/Sabrinayusrina03/eksperimen_SML_SabrinaYusrina.mlflow
ENV MLFLOW_TRACKING_USERNAME=Sabrinayusrina03

# entry point container
ENTRYPOINT ["mlflow", "run", "MLProject", "--experiment-name", "Docker_CI_Run"]