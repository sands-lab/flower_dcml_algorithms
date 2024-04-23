# Framework Usage

# Introduction

This repository contains a set of FL algorithms that support model customization (i.e. different clients have different model architectures) or split learning.

The framework is easy to extend: you just need to include the server logic in a python script in the `src/strategies` folder, the client logic in a script in the `src/clients` folder, and a configuration file in the `config/fl_algorithm` folder. Check out existing algorithms to see the provided level of abstraction and to find quite a bit of re-usable logic (decorators, functions, methods, ...), which may help you so as not to write too much flower boilerplate code.


## Data preparation

Any data preparation (download, partitioning, distribution to clients, ...) needs to be done off-line, i.e. before running the actual FL experiment. This is done so as not to introduce any computational requirements to the clients when deploying the algorithms on the testbed.

To download the data and create the client dataset-subsets, you may use the `generate_partitions.py` script. This script automatically downloads the data, saves it in the required format, and generates the files that determine the partitioning. For instance:

```bash
python generate_partitions.py \
    --dataset_name=cifar10 \
    --n_clients=50 \
    --seed=1602 \
    --test_percentage=0.2 \
    --partitioning_method=iid \
    --fixed_training_set_size=200 \
    --val_percentage=0.1
```

generates a `data/partitions/cifar10/iid_50clients_1602seed_0.2test_0.1val_0holdoutsize_200trainsize` folder. This contains a `generation_config.json` file with the configuration stored in a `.json` file, and three `.csv` files for every client:

- a `partition_X_train.csv`, where *X* is the sequence number of the client;
- a `partition_X_test.csv`, where *X* is the sequence number of the client.
- a `partition_X_val.csv`, where *X* is the sequence number of the client.

Note, that the script requires the path, where should the raw data be stored, to be set as an environment variable, `FLTB_DATA_HOME_FOLDER`. This value should be constant across all experiments so as not to store the same data points multiple times. The value of the environment variable can be set in the `.env` configuration file when running the test on a simulation environment, while it should be set by the runtime when running the actual experiment on the testbed.

For further details regarding how to run the script, you may run:

```bash
python generate_partitions.py --help
```


## Testing the algorithms

To simulate the experiment you just need to issue the command `python fl.py`. As above, the `FLTB_DATA_HOME_FOLDER` needs to be set. The script will load the configuration from `config/hydra/base_config.yaml` file, which includes:

- `dataset`: name of the dataset to be used;
- `partitioning_configuration` i.e. folder, where are the partition-related data stored, e.g. the `.csv` files with data about the splitting. In other words, this parameter is the folder created by the `generate_partitions.py` script.
- `fl_algorithm`, which indicates both the strategy and the client functions of the FL algorithm;
- Any data concerning local training and global training.

Note, that `dataset` and `partitioning_configuration` uniquely determine the data partitioning configuration - during runtime, the clients will have data as set in the `data/partitions/{dataset}/{partitioning_configuration}` folder.

To run an experiment with different configuration, you need to override the default configuration in `config/hydra/base_config.yaml`. Here's a few examples how to achieve this:

```bash
python fl.py global_train.epochs=20  # run the algorithm for 20 global epochs
./run_experiment fl_algorithm=fedprox  # use the fedprox algorithm for training
```

Refer to hydra documentation for further details on overriding configuration files.


When running the algorithm on the testbed (or indeed, whenever you want to separately run the client and the server), use the `run_server.py` and the `run_client.py` scripts.

### Running on docker

In order to run the experiment on docker, use the following commands:

```bash
docker build -t fl_base_image .
docker run -it --rm -v ./data/raw/:/app/data/raw fl_base_image
```


## Algorihtms

Currently the following algorithms are implemented:

#### Private training

Simulation of the accuracy achieved if every client trains independently on its own dataset:

```bash
python fl.py fl_algorithm=private_training
```

#### FedAvg

```bash
python fl.py fl_algorithm=fedavg
```


#### FedProx

```bash
python fl.py fl_algorithm=fedprox
```


#### FD

Implementation of the algorihtm proposed `Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data`.

```bash
python fl.py fl_algorithm=fd
```

#### DS-FL (FedMD)

Implementation of `Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training With Non-IID Private Data` and `FedMD: Heterogenous Federated Learning via Model Distillation`

```bash
python fl.py fl_algorithm=ds_fl
```

#### Lg-FedAvg

Implementation of `Think Locally, Act Globally: Federated Learning with Local and Global Representation`.

```bash
python fl.py fl_algorithm=lg_fedavg
```


#### FedDF

Implementation of `Ensemble Distillation for Robust Model Fusion in Federated Learning`

```bash
python fl.py fl_algorithm=feddf
```


#### HeteroFL

Implementation of `HETEROFL: COMPUTATION AND COMMUNICATION EFFICIENT FEDERATED LEARNING FOR HETEROGENEOUS CLIENTS`

```bash
python fl.py fl_algorithm=heterofl
```


#### Federated Dropoud

Implementation of `EXPANDING THE REACH OF FEDERATED LEARNING BY REDUCING CLIENT RESOURCE REQUIREMENTS`

```bash
python fl.py fl_algorithm=federated_dropout
```


#### FedKD

Implementation of `FedKD: Communication Efficient Federated Learning via Knowledge Distillation`

```bash
python fl.py fl_algorithm=fedkd
```


#### Split Learning

In this case, you need to use the `sl.py` script (no changes required to the `run_client.py` and `run_server.py` scripts)

```bash
python sl.py fl_algorithm=split_learning
```


## Logging data to W&B

To log the data to wandb you need to:

- Set the environment variable `LOG_TO_WANDB` to `1`;
- Create a `secrets.env` file with the following structure:

```bash
WANDB_API_KEY=<fill your value>
WANDB_USERNAME=<fill your value>
WANDB_ENTITY=<fill your value>
WANDB_PROJECT=<fill your value>
```


## Model customization

In several algorithms, each client may independently choose its model architecture (restrictions apply depending on the algorithm, e.g. in PerFed all model architectures need to share the same architecture for the lowermost layers). In the implementation, this is achieved by assigning to every client an integer value $C$ which states its capacity. In the testbed, this value is configured manually, while in the simulations (`fl.py`) this value is set randomly.

Either way, model customization is configured in the following `.json` files:

- `config/colext/device_capacities.json`: configure the mapping device type - capacity tier;
- `config/models/*`: configuration for all the models used by the algorithms.
