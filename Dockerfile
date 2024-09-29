## 推荐使用nvidia-pytorch镜像，目前自带flash-attn的基础依赖环境，只需要后续进行版本适配即可
# FROM nvcr.io/nvidia/pytorch:23.10-py3
FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set global.extra-index-url "https://pypi.doubanio.com/simple"

## opencv-related bug: thanks to: https://soulteary.com/2024/01/07/fix-opencv-dependency-errors-opencv-fixer.html
# RUN pip install opencv-fixer==0.2.5
# RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN pip install opencv-python=='4.7.0.72'

## 这里建议从本地build xtuner，方便支持自定义的dataset等
# RUN pip install -U 'xtuner[deepspeed]'
# RUN git clone https://github.com/internlm/xtuner.git
WORKDIR /xtuner
COPY . /xtuner/
RUN pip install --no-cache-dir -e .['deepspeed']

RUN pip install packaging
## https://github.com/InternLM/xtuner/issues/744 which flash-attn==2.3.6 is recommend
## but transformer-engine 1.2.1+bbafb02 requires flash-attn!=2.0.9,!=2.1.0,<=2.3.3,>=1.0.6
## docker build的时候可能卡死，最后这一步可以在docker内执行
RUN MAX_JOBS=4 pip install flash-attn==2.3.0 --no-build-isolation
