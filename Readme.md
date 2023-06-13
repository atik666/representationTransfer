# Unsupervised Representation Learning to Aid Few-Shot Transfer Learning
Pytorch implementation of Model Agnostic Meta Learning (MAML)

## Environment Preparing
Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed on your device
```bash
$ conda create -n maml-impl
$ conda activate maml-impl
```
Install [Cuda 11.7](
https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=runfile_local)

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run

$ sudo sh cuda_11.7.1_515.65.01_linux.run
```

Install [pytorch](https://pytorch.org/) based off Cuda 11.7


Install remaining packages through pip:

```
pip install -r requirement.txt
```

# Data preparation

Datasets used: 

* mini-imagenet
* Omniglot

**Refer to notebooks/Datasets Download.ipynb to Download mini-imagenet and omniglot data for project**

# Mini-Imagenet
To appply mini-imagenet with MAML change directory

```shell
cd miniImagenet
```

Dowanload mini-imagenet dataset and put it in dataset folder.

```shell
cd dataset
```

it contains folders like this:

```shell
test/
train/
unsupervised/
val/
```

To get the unlabeled data in the unsupervised folder, run:

```python
python moving_files.py
```

To run the unsupervised representation learning model, run:

```python
python miniimagenet_transfer.py
```
*Note: The model will be save in saved_models folder.*

To run the supervised transfer learning model, run:

```python
python mminiimagenet_main.py
```

To change the parameters, run:

```python
python miniimagenet_main.py --n_way=20, --k_shot=5, --k_query=5
```
To perfrom regular MAML without transfer learning, run: 
```python
python miniimagenet_main.py --transfer='False'
```

# Omniglot

To appply Omniglot with MAML change directory

```shell
cd Omniglot
```

Dowanload Omniglot dataset and put it in dataset folder.

```shell
cd dataset
```

To run the unsupervised representation learning model, run:

```python
python omniglot_transfer.py
```
*Note: The model will be save in saved_models folder.*


To run the supervised transfer learning model, run:

```python
python omniglot_main.py
```

To perfrom regular MAML without transfer learning, run: 
```python
python omniglot_main.py --transfer='False'
```
