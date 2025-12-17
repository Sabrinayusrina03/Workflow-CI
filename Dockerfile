FROM continuumio/miniconda3

WORKDIR /app

# Salin semua file proyek
COPY . /app

ARG DAGSHUB_TOKEN

# Instal dependencies
RUN conda install -c conda-forge --name base python=3.12.7 pip && \
    pip install mlflow dagshub && \
    echo "Dependencies installed successfully"

# Set environment variables
ENV MLFLOW_TRACKING_URI=https://dagshub.com/Sabrinayusrina03/eksperimen_SML_SabrinaYusrina.mlflow
ENV MLFLOW_TRACKING_USERNAME=Sabrinayusrina03
ENV MLFLOW_TRACKING_PASSWORD=${DAGSHUB_TOKEN}

# entry point container
ENTRYPOINT ["mlflow", "run", "MLProject", "--experiment-name", "Docker_CI_Run"]