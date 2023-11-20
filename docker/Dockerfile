# Start from a core stack version
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN pip install --no-cache-dir -U pip
RUN apt-get update && apt-get install -y git git-lfs libglib2.0-0 libgl1-mesa-dev libgl1-mesa-glx

# Clone your project
RUN git clone https://github.com/internlm/xtuner.git


# Install the requirements in the docker container
RUN cd xtuner && pip install --no-cache-dir -e .['deepspeed']
