# NeMo-Megatron框架运行Llama2 1B模型

## 系统配置

T集群，Slurm裸机环境

| Software                | Version              |
|-------------------------|----------------------|
| GCC                     | 10.2.0               |
| CUDA                    | 11.8                 |
| cuDNN                   | 8.9.5.29             |

- 非root权限安装`cuda 11.8`，参考[这里](https://github.com/pyg-team/pytorch_geometric/issues/392#issuecomment-503335625)
- 非root权限安装`cudnn 8.9.5.29`，参考[这里](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)
- 安装`apex`，要求`GCC`的大版本号大于5，且小于12
- 安装`transformer_engine`需要`cuda >= 11.8`，参考[这里](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html)
- `GCC/CUDA/cuDNN`用于编译安装`apex`, `transformer_engine`, `flash-attn`

配置环境变量
```
# GCC可以使用T集群share_data里面的
# CUDA和cuDNN可以自己安装在$HOME/dev路径下
export GCC_HOME=/mnt/petrelfs/share_data/llm_env/dep/gcc-10.2.0
export MPFR_HOME=/mnt/petrelfs/share_data/llm_env/dep/mpfr-4.1.0
export CUDA_HOME=$HOME/dev/cuda_11.8
export PATH=$GCC_HOME/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:${MPFR_HOME}/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include
export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/c++
```

## 环境搭建

```
mkdir -p $HOME/dev
export WORKSPACE=$HOME/dev

cd $WORKSPACE
git clone https://github.com/pjlab-sys4nlp/NeMo.git
cd NeMo
export NEMO_DIR=$PWD
conda create --name nemo-megatron python=3.10 -y
conda activate nemo-megatron

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install packaging ninja
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_common.txt
pip install -r requirements/requirements_lightning.txt
pip install -r requirements/requirements_nlp.txt

cd $WORKSPACE
git clone https://github.com/pjlab-sys4nlp/NeMo-Megatron-Launcher.git
cd NeMo-Megatron-Launcher
export NEMO_MEGATRON_DIR=$PWD
pip install -r requirements.txt

cd $WORKSPACE
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext --distributed_adam --deprecated_fused_adam

# 安装transformer_engine的时候会将flash-attn作为依赖安装
cd $WORKSPACE
git clone https://github.com/NVIDIA/TransformerEngine.git --recurse-submodules
cd TransformerEngine
git checkout stable
export NVTE_FRAMEWORK=pytorch
pip install .

cd $WORKSPACE
git clone https://github.com/pjlab-sys4nlp/Megatron-LM.git
cd Megatron-LM
pip install .
```

## 模型并行配置以及超参数配置

配置信息定义在`launcher_scripts/conf/config.yaml`以及`launcher_scripts/conf/training/llama2_1b.yaml`

注意：`launcher_scripts/conf/config.yaml`中的`launcher_scripts_path`需要修改为实际路径

## 运行Llama2模型

```
cd $NEMO_MEGATRON_DIR/launcher_scripts
conda activate nemo-megatron
python main.py
```

## 日志信息以及TensorBoard数据

日志以及tb路径

| Name                    | Path                                             |
|-------------------------|--------------------------------------------------|
| Log                     | launcher_scripts/results/llama2_1b               |
| TensorBoard             | launcher_scripts/results/llama2_1b/results       |
