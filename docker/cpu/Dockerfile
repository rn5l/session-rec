FROM continuumio/miniconda3

WORKDIR /app

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment_cpu.yml .
RUN conda env create -f environment_cpu.yml

# Initialize conda in bash config fiiles:
RUN conda init bash

# Activate the environment, and make sure it's activated:
RUN echo "conda activate srec37" > ~/.bashrc
RUN echo "Make sure flask is installed:"
RUN python -c "import tensorflow as tf"

# The code to run when container is started:
#COPY python.sh .
#ENTRYPOINT ["/bin/bash"]