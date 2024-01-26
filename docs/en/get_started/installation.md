### Installation

In this section, we will show you how to install XTuner.

- It is recommended to build a Python-3.10 virtual environment using conda

  ```bash
  conda create --name xtuner-env python=3.10 -y
  conda activate xtuner-env
  ```

- Install XTuner via pip

  ```shell
  pip install -U xtuner
  ```

  or with DeepSpeed integration

  ```shell
  pip install -U 'xtuner[deepspeed]'
  ```

- Install XTuner from source

  ```shell
  git clone https://github.com/InternLM/xtuner.git
  cd xtuner
  pip install -e '.[all]'
  ```
