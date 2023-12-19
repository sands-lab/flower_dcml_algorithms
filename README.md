# Framework Usage

## Data preparation

Any data preparation (download, partitioning, distribution to clients, ...) needs to be done off-line, i.e. before running the actual FL experiment. This is done so as not to introduce any computational requirements to the clients.

To download the data and create the partitionings, you may use the `generate_partitions.py` script. This script automatically downloads the data, saves it in the required format, and generates the files that determine the partitioning. For instance:

```bash
python generate_partitions.py \
    --dataset_name=cifar10 \
    --num_clients=50 \
    --seed=1602 \
    --test_percentage=0.2 \
    --partition_method=dirichlet \
    --alpha=50 \
    --min_size_of_dataset=10
```

generates a `data/partitions/cifar10/dirichlet_50clients_1602seed_50.0alpha_0.2test` folder. This contains a `generation_config.json` file with all the configuration stored in a `.json` format, and two files for every client:

- a `partition_X_train.csv`, where *X* is the sequence number of the client;
- a `partition_X_test.csv`, where *X* is the sequence number of the client.

Note, that the script requires the path, where should the raw data be stored, to be set as an environment variable, `FLTB_DATA_HOME_FOLDER`. This value should be constant across all experiments so as not to store the same data points multiple times. The value of the environment variable can be set in the `.env` configuration file when running the test on IBEX, while it should be set by the runtime when running the actual experiment on the testbed.

For further details regarding how to run the script, you may run:

```bash
python generate_partitions.py --help
```


## Testing the algorithms on IBEX

To simulate the experiment on IBEX (or locally), you just need to issue the command `python fl.py`. As above, the `FLTB_DATA_HOME_FOLDER` needs to be set. The script will load the configuration from `conf/base_config.yaml` file, which includes:

- `dataset`: name of the dataset to be used;
- `partitioning_configuration` i.e. folder, where are the partition-related data stored, e.g. the `.csv` files with data about the splitting. In other words, this parameter is the folder created by the `generate_partitions.py` script.
- `fl_algorithm`, which indicates both the strategy and the client functions of the FL algorithm;
- Any data concerning local training and global training.

Note, that `dataset` and `partitioning_configuration` uniquely determine the data partitioning configuration - during runtime, the clients will have data as set in the `data/partitions/{dataset}/{partitioning_configuration}`.

Alternatively, you may also run the `./run_experiment.sh` command (**recommended approach**).

To run an experiment with different configuration, you need to override the default configuration in `conf/base_config.yaml`. Here's a few examples how to achieve this:

```bash
python fl.py global_train.epochs=20  # run the algorithm for 20 global epochs
./run_experiment fl_algorithm=fedprox  # use the fedprox algorithm for training
```

Refer to hydra documentation for further details on overriding configuration files.

### Configuring environment variables

The `run_server.sh` and the `run_experiment.sh` scripts may be customized with some variables, which are set within the script:

* `LOG_TO_WANDB`: if set to `1`, the accuracy will be persisted and logged to a local folder within the `logs` folder. If set to `0`, the accuracy will only be printed to the output.
* `SYNC_WITH_WANDB_CLOUD`: if set to `1`, after the experiment the results will be synchronized with the wandb cloud. If set to `0`, the results will only be available locally.
* `IBEX_SIMULATION`: set to `1` if running on IBEX, locally, or on docker. Setting the value to `1` causes the environment variables to be read from the `.env` file. When running on the testbed, all the environment variables should be set by the runtime, so in this case you should set the variable to `0`.

### Running on docker

In order to run the experiment on docker, use the following commands:

```bash
docker build -t fl_base_image .
docker run -it --rm -v ./data/raw/:/app/data/raw fl_base_image
```


## Algorihtms

Currently the following algorithms are implemented:

#### Private training

Simulation of the accuracy achieved if every client trained independently on its own dataset:

```bash
python fl.py fl_algorithm=private_training
```

*Comments*: this procedure is still implemented in Flower. This means, that at every epoch a sample of clients are sampled and trained with the parameters set in `base_config.yaml`.

#### FedAvg

```bash
python fl.py fl_algorithm=fedavg
```

*Comments*: state which model architecture (by means of the `client_capacity` parameter) in the `fedavg.yaml` file.


#### FedProx

```bash
python fl.py fl_algorithm=fedprox
```

*Comments*: Set the regularization strength and the model architecture in the `fedprox.yaml` file.


#### FD

Implementation of the algorihtm proposed `Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data`.

```bash
python fl.py fl_algorithm=fedprox
```

*Comments*: Set the regularization strength in the `fedprox.yaml` file.

### Work in progress

#### FedMD

Implementation of the algorithm proposed in `FedMD: Heterogenous Federated Learning via Model Distillation`

```bash
python fl.py fl_algorithm=fedmd
```

*Comments*: Paper lacks a lot of details regarding the implementation. For instance:

- it is not stated which loss function is used for distilling knowledge. The authors only state that `the models communicate and align their logits computed from public data without applying the softmax activation layer`. In the implementation, I decided to use MSE loss.
- it is not described how to switch from the pre-trained model to the actual model of interest. This is something that still needs to be implemented in our code.
- **We should drop this algorithm!!**


#### DS-FL

Implementation of `Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training With Non-IID Private Data`

```bash
python fl.py fl_algorithm=ds_fl
```

*Comments*:

- In the paper, it seems that every clients needs to participate in every training epoch (double check this). In the implementation, we support partial client participation;
- TO-DO: Currently, the clients do not share the index set of the data points for the next training epoch. Instead, the whole public dataset is exchanged at every training epoch.


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



## TO-DOs

- Check all the algorithms implemented so far for correctness;
- Correct FedGKT;
- Try to reproduce some results;
- Next implement: FedKD, HeteroFL/FjORD, FedRolex, Federated Dropout, FedRecon;
- Run experiment on testbed.