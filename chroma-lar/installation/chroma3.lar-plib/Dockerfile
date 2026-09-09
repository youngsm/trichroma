FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/
ADD https://api.github.com/repos/youngsm/chroma-lite/git/refs/heads/master /opt/chroma_version.json
RUN git clone https://github.com/youngsm/chroma-lite
RUN mv chroma-lite chroma
RUN sed -i 's/VIRTUAL_ENV/CONDA_PREFIX/g' setup.py #use anaconda env instead

# quick fix for EventViewer
RUN sed -i '741s/self\.i = len(self\.l-1)/self.i = len(self.l)-1/' chroma/camera.py
RUN python3 -m pip install -e .
# below are extras for analyses
RUN python3 -m pip install -U trimesh numpy scipy scikit-learn h5py fire scikit-build
RUN python3 -m pip install git+https://github.com/CIDeR-ML/photonlib.git
RUN python3 -m pip install git+https://github.com/CIDeR-ML/siren-lartpc.git
RUN python3 -m pip install gdown
RUN python3 -m pip install rich
WORKDIR /workspace/
RUN git clone https://github.com/youngsm/chroma-lar
WORKDIR /workspace/chroma-lar/
RUN sh chroma_lar_env.sh
RUN python3 -m pip install -e .
WORKDIR /workspace/
RUN python3 -m pip install git+https://github.com/CIDeR-ML/sirentv.git

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
