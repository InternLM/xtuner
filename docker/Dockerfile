FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install opencv-python=='4.7.0.72'

RUN git clone https://github.com/internlm/xtuner.git

RUN cd xtuner && pip install --no-cache-dir -e .['deepspeed']
