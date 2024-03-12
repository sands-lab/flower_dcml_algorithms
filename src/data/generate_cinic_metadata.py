import os
import json

from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd


def main():
    load_dotenv()
    raw_data_folder = os.getenv("COLEXT_DATA_HOME_FOLDER")
    cinic_data_folder = os.path.join(raw_data_folder, "cinic")

    # define mapping from name class to index
    class_mapping = {class_name: i for i, class_name in enumerate([
        f for f in os.listdir(cinic_data_folder) if os.path.isdir(f"{cinic_data_folder}/{f}")
    ])}
    assert len(class_mapping) == 10
    with open(f"{cinic_data_folder}/class_mapping.json", "w") as fp:
        json.dump(class_mapping, fp, indent=4)

    out_dict = {"filename": [], "label": []}
    skipped_images = 0
    for label, idx in class_mapping.items():
        label_folder = f"{cinic_data_folder}/{label}/"
        for file in tqdm(os.listdir(label_folder), desc=label):
            if Image.open(f"{cinic_data_folder}/{label}/{file}").mode == "RGB":
                out_dict["filename"].append(f"{label}/{file}")
                out_dict["label"].append(idx)
            else:
                skipped_images += 1
    print(f"Skipped {skipped_images} images")
    pd.DataFrame(out_dict).to_csv(f"{cinic_data_folder}/metadata.csv", index=False)


if __name__ == "__main__":
    main()
