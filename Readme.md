# Unsupervised Representation Learning to Aid Few-Shot Transfer Learning
pytorch implementation of MAML

# Data preparation

datasets used: 

* mini-imagenet
* Omniglot

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

To get the dataset in the desired style, download from here: 

https://drive.google.com/file/d/19a_6Krjjv3XCQwiZGqFtTdGb4KHMFl21/view?usp=sharing

unzip the file and put the contents in dataset folder.

To get the unlabeled data in the unsupervised folder, run:

```python
python moving_files.py
```

To run the unsupervised representation learning model, run:

```python
python miniimagenet_transfer.py
```
The model will be save in saved_models folder.

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

############################################################

To appply Omniglot with MAML change directory

```shell
cd Omniglot
```

Dowanload Omniglot dataset and put it in dataset folder.

```shell
cd dataset
```

To get the dataset in the desired style, download from here: 

https://drive.google.com/file/d/1A_A8Huh4os6zzot8GMD5vBSqHc3wX9Fk/view?usp=sharing

To run the unsupervised representation learning model, run:

```python
python omniglot_transfer.py
```
The model will be save in saved_models folder.

To run the supervised transfer learning model, run:

```python
python omniglot_main.py
```

To perfrom regular MAML without transfer learning, run: 
```python
python omniglot_main.py --transfer='False'
```
