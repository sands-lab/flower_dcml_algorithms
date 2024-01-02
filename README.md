# Framework Usage

## Data preparation

Any data preparation (download, partitioning, distribution to clients, ...) needs to be done off-line, i.e. before running the actual FL experiment. This is done so as not to introduce any computational requirements to the clients.

To download the data and create the partitionings, you may use the `generate_partitions.py` script. This script automatically downloads the data, saves it in the required format, and generates the files that determine the partitioning. For instance:

```bash
python generate_partitions.py \
    --dataset_name=cifar10 \
    --n_clients=50 \
    --seed=1602 \
    --test_percentage=0.2 \
    --partitioning_method=dirichlet \
    --alpha=50 \
    --min_size_of_dataset=10
```

generates a `data/partitions/cifar10/dirichlet_50clients_1602seed_50.0alpha_0.2test` folder. This contains a `generation_config.json` file with the configuration stored in a `.json` file, and two `.csv` files for every client:

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

Note, that `dataset` and `partitioning_configuration` uniquely determine the data partitioning configuration - during runtime, the clients will have data as set in the `data/partitions/{dataset}/{partitioning_configuration}` folder.

Alternatively, you may also run the `./run_experiment.sh` command (**recommended approach**).

To run an experiment with different configuration, you need to override the default configuration in `conf/base_config.yaml`. Here's a few examples how to achieve this:

```bash
python fl.py global_train.epochs=20  # run the algorithm for 20 global epochs
./run_experiment fl_algorithm=fedprox  # use the fedprox algorithm for training
```

Refer to hydra documentation for further details on overriding configuration files.

### Configuring environment variables

The `run_server.sh` and the `run_experiment.sh` scripts may be customized with some variables, which are set within the script:

* `LOG_TO_WANDB`: if set to $1$, the accuracy will be persisted and logged to a local folder within the `logs` folder. If set to $0$, the accuracy will only be printed to the output.
* `SYNC_WITH_WANDB_CLOUD`: if set to $1$, after the experiment the results will be synchronized with the wandb cloud. If set to $0$, the results will only be available locally.
* `IBEX_SIMULATION`: set to $1$ if running on IBEX, locally, or on docker. Setting the value to $1$ causes the environment variables to be read from the `.env` file. When running on the testbed, all the environment variables should be set by the runtime, so in this case you should set the variable to $0$.

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

*Comments*:

- this procedure is implemented in Flower. This means, that at every epoch, a sample of clients are sampled and trained with the parameters set in `base_config.yaml`. When evaluating the model, a random sample of clients is selected. The selected clients may or may not have been trained in the last epoch. If this is something you wish to avoid, set `global_train.fraction_fit` to `1.0`, so that every client is trained at every epoch.

#### FedAvg

```bash
python fl.py fl_algorithm=fedavg
```

*Comments*:

- You can state which model architecture should be used in the `fedavg.yaml` file by setting the `client_capacity` parameter.


#### FedProx

```bash
python fl.py fl_algorithm=fedprox
```

*Comments*:

- Set the regularization strength and the model architecture in the `fedprox.yaml` file.
- Same consideration as for FedAvg regarding model architecture.

#### FD

Implementation of the algorihtm proposed `Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data`.

```bash
python fl.py fl_algorithm=fd
```

*Comments*:

- Set the regularization strength in the `fd.yaml` file;
- *the uploaded local-average logit vectors from all devices are averaged, resulting in a global-average logit vector per label*, but in the pseudocode the the output logits are never averaged. Also, the output logits are not "global", but are personalized for every client;
- It is not stated explicitly in the text, but the algorithm requires full client participation;
- Uses CE as the distillation loss function;
- the temperature for the softmax function is never addressed, so I assumed it was set to $1$. Nevertheless, this parameter may be changed in the `fd.yaml` file;
- In the paper, also FedAug is proposed, but we ignore this extension in the implementation;
Reproducing the paper results:

*Reproducing the results*

- Not stated model architecture, but only the number of parameters. I managed to obtain the same number of parameters by using the architecture called `Net` in https://github.com/pytorch/examples/blob/main/mnist/main.py and by setting all the biases to `False`.
- The way they construct the non-IID datasets is not completely clear;
- Not stated which optimizer they used (SGD, Adam, ...), nor any hyperparameter (learning rate, strength of the KD regularization term, ...)
- From the algorithm, it is not clear how to handle cases in which some client does not have some target labels. Line 8 will result in an exception. In the paper *Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training With Non-IID Private Data* they propose a solution, but it does not seem right (denominator should have -1 or not depending on whether the client has label or not);
- could not reproduce the results, probably because different hyperparameters, model architecture etc.
- The results I obtain are much better than the ones reported in the paper;
- The algorithm as implemented outperforms private training.

#### DS-FL

Implementation of `Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training With Non-IID Private Data`

```bash
python fl.py fl_algorithm=ds_fl
```

*Comments*:

- From the paper, it is not clear whether every clients needs to participate in every training epoch (*the evaluations are conducted without any missing clients per round*). In the implementation, we support partial client participation under the condition, that the same set of clients is sampled during the update and distillation phase in algorithm 1.;
- In the implementation, the clients do not share the index set of the data points for the next training epoch. Instead, the whole public dataset is exchanged at every training epoch.

#### Lg-FedAvg

Implementation of `Think Locally, Act Globally: Federated Learning with Local and Global Representation`.

```bash
python fl.py fl_algorithm=lg_fedavg
```

*Comments*:

- In section C.2.2 the authors say that:
    - They resize the input images to 224x224, but this doesn't make sense! Besides, in the code they don't do it https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py;
    - `We use the two convolutional layers for the global model` but this is just the opposite of what they are doing! In the paper they propose to use the lowermost layers as private model and uppermost as public model!
- In the repository, they say that in order to reproduce the results, you first need to run FedAvg and then load the model obtained with FedAvg and further train it with LG-FedAvg, but in the paper they never say they do anything of the sort. In the paper they only say in section C.2.2 that *We train LG-FEDAVG with global updates until we reach a goal accuracy (57% for CIFAR-10) before training for additional rounds to jointly update local and global models.*.....
- In section C.2.1 they say they use the last two layers to form the global model, but in order to get the stated number of parameters you need to take the last $3$ layers.

#### PerFed

Implementation of `Federated Learning with Personalization Layers`.

```bash
python fl.py fl_algorithm=perfed
```

#### FedRecon

Implementation of `Federated Reconstruction: Partially Local Federated Learning`


```bash
python fl.py fl_algorithm=fedrecon
```


#### FedDF

Implementation of `Ensemble Distillation for Robust Model Fusion in Federated Learning`

```bash
python fl.py fl_algorithm=feddf
```

### Work in progress

#### FedKD

Implementation of `FedKD: Communication Efficient Federated Learning via Knowledge Distillation`

```bash
python fl.py fl_algorithm=fedkd
```

*Comments*:

- The way it is written `Algorithm 1` in the paper, it seems that gradients are synchronized after each GD update. However, this seems very inefficient (also considering that the paper is titled "communication efficient"), so we resolve to use a more practical approach, i.e. training for a given number of local training epochs;
- In the paper, the authors propose SVD & encription to reduce communication and to increase security. However, security is not the focus of this repository, so we omit it. Also, implementing SVD only for this algorithm would give to this algorithm an unfair advantage over the others. Therefore, in the implementation we omit SVD-ing the gradients.


#### FedGKT

Implementation of `Group Knowledge Transfer:
Federated Learning of Large CNNs at the Edge`

```bash
python fl.py fl_algorithm=fedgkt
```

*Comments*:

- Implementation contains some bug

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

In several algorithms, each client may independently choose its model architecture (restrictions apply depending on the algorithm, e.g. in PerFed all model architectures need to share the same architecture for the lowermost layers). In the implementation, this is achieved by assigning to every client an integer value $C$ which states its capacity. In the testbed, this value will need to be configured manually, while in the simulations (`fl.py`) this value is set randomly.

Either way, after determining the model capacity, the client loads the model as specified in the `model_mapping.json` file.

## TO-DOs

- Check all the algorithms implemented so far for correctness;
- Correct FedGKT;
- Try to reproduce some results;
- Next implement: FedKD, HeteroFL/FjORD, FedRolex, Federated Dropout;
- Run experiment on testbed.