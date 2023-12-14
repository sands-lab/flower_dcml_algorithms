# Usage

## Data preparation

Any data preparation (download, partitioning, distribution to clients, ...) needs to be done off-line, i.e. before running the actual FL experiment. This is done so as not to introduce any computational requirements to the clients.

To download the data and create the partitionings, you may use the `generate_partitions.py` script. This script automatically downloads the data, saves them in the required format, and generates the files that determine the partitioning. For instance:

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

generates a `data/partitions/cifar10/dirichlet_50clients_1602seed_50.0alpha_0.2test` folder. This contains a `generation_config.json` file with all the configuration stored in a `.json` format, and two filtes for every client:

- a `partition_X_train.csv`, where *X* is the sequence number of the client;
- a `partition_X_test.csv`, where *X* is the sequence number of the client;

Note, that the script requires the path, where should the raw data be stored, to be set as an environment variable `FL_DATA_HOME_FOLDER`. This value should be constant across all experiments so as not to store the same data points multiple times. The value of the environment variable can be set in the `.env` configuration file when running the test on IBEX, while it should be set by the runtime when running the actual experiment on the testbed.

For further details regarding how to run the script, you may run:

```bash
python generate_partitions.py --help
```


## Testing the algorithms on IBEX

To simulate the experiment on IBEX (or locally), you just need to issue the command `python fl.py`. As above, the `FL_DATA_HOME_FOLDER` needs to be set. The script will load the configuration from `conf/base_config.yaml` file, which includes:

- `dataset`: name of the dataset to be used;
- `partitioning_configuration` i.e. folder, where are the partition-related data stored, e.g. the `.csv` files with data about the splitting. In other words, this parameter is the folder created by the `generate_partitions.py` script.
- `fl_algorithm`, which indicates both the strategy and the client functions of the FL algorithm;
- Any data concerning local training and global training.

Note, that `dataset` and `partitioning_configuration` uniquely determine the data partitioning configuration - during runtime, the clients will have data as set in the `data/partitions/{dataset}/{partitioning_configuration}`.

Alternatively, you may also run the `./run_experiment.sh` command.