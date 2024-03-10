import torchvision.transforms as T

from src.data.cv_dataset import CustomDataset
from src.helper.commons import read_json
from src.data.dataset_partition import DatasetPartition


def init_dataset(
    cid,
    dataset_partition,
    dataset_name,
    partition_folder,
    images_folder,
):
    norm_config = read_json("config/data/data_configuration.json",
                            [dataset_name, "normalization_parameters"])
    transforms = [
        T.ToTensor(),
        T.Normalize(mean=norm_config["mean"], std=norm_config["std"])
    ]

    is_train = dataset_partition == DatasetPartition.TRAIN
    if is_train and dataset_name not in {"mnist"}:
        transforms.append(T.RandomHorizontalFlip())
    transforms = T.Compose(transforms)

    filename = f"partition_{cid}_{dataset_partition.value}.csv"
    partition_file = f"{partition_folder}/{filename}"
    dataset = CustomDataset(
        images_folder,
        partition_csv=partition_file,
        transforms=transforms
    )
    return dataset
