# Use the official Python 3.9 image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Install system dependencies such as ffmpeg
RUN apt-get update && \
    apt-get install -y software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
    python3.9 \
    python3-pip \
    g++ \
    make \
    cmake \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the local project files to the Docker image
COPY . /app

# Install PyTorch, torchvision and torchaudio, with CUDA supported
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyTorch3D directly from the GitHub repository
RUN wget https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.4.zip
RUN unzip v0.7.4.zip
RUN cd pytorch3d-0.7.4/ && python setup.py install

# Install MMCV with openmim and other dependencies
RUN pip install cython
RUN pip install openmim==0.3.9
RUN mim install mmcv==2.1.0

# Install other dependencies from the provided requirements file
RUN pip install -r app/docs/prepare_env/requirements.txt --use-deprecated=legacy-resolver

# Expose port 8000 (or whichever port your app uses)
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
