# Lab 1

The code contains the implementation of the BC, GAIL, DAgger, FEM, MWAL, MBRL_BC, MBRL_GAIL.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1leDW3IzeM83R3xgql6o22qQUBVnS9pxO/view?usp=sharing)

## New parts

### adding the folder `gail1` with trpo replaced by `td3`

* Draw a figure like this
![Figure3](../static/Snipaste_2022-05-06_17-02-00.png)
* The figure in the original GAIL
![Figure4](../static/Snipaste_2022-05-13_07-02-53.png)

<hr>

## Old Parts

### Requirements

We use Python 3.6 to run all experiments. Please install MuJoCo following the instructions from [mujoco-py](https://github.com/openai/mujoco-py). Other python packages are listed in [requirement.txt](requirement.txt)

### Dataset

Dataset, including expert demonstrations and expert policies (parameters), is provided in the folder of [dataset](dataset).

However, one can run SAC to re-train expert policies (see [scripts/run_sac.sh](scripts/run_sac.sh)) and to collect expert demonstrations (see [scripts/run_collect.sh](scripts/run_collect.sh)).

### Usage

The folder of [scripts](scripts) provides all demo running scripts to test algorithms like GAIL, BC, DAgger, FEM, GTAL, and imitating-environments algorithms.



