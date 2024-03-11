### Installation

In this section, we will show you how to install XTuner.

## Installation Process

We recommend users to follow our best practices for installing XTuner.
It is recommended to use a conda virtual environment with Python-3.10 to install XTuner.

### Best Practices

**Step 0.** Create a Python-3.10 virtual environment using conda.

```shell
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env
```

**Step 1.** Install XTuner.

Case a: Install XTuner via pip:

```shell
pip install -U xtuner
```

Case b: Install XTuner with DeepSpeed integration:

```shell
pip install -U 'xtuner[deepspeed]'
```

Case c: Install XTuner from the source code:

```shell
git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'
# "-e" indicates installing the project in editable mode, so any local modifications to the code will take effect without reinstalling.
```

## Verify the installation

To verify if XTuner is installed correctly, we will use a command to print the configuration files.

**Print Configuration Files:** Use the command `xtuner list-cfg` in the command line to verify if the configuration files can be printed.

```shell
xtuner list-cfg
```

You should see a list of XTuner configuration files, corresponding to the ones in [xtuner/configs](https://github.com/InternLM/xtuner/tree/main/xtuner/configs) in the source code.
