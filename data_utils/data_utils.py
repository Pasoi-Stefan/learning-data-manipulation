from .text_data_processors import get_data as get_text_data
from .image_data_processors import get_data as get_image_data


import pandas as pd
from .text_data_processors import InputExample

def get_covid19stance_data():
    examples = {
        "train" : [],
        "dev": [],
        "test": []
    }

    train_examples = pd.read_csv("datasets/covid-19-stance/raw_train_all_onecol.csv")
    dev_examples = pd.read_csv("datasets/covid-19-stance/raw_val_all_onecol.csv")
    test_examples = pd.read_csv("datasets/covid-19-stance/raw_test_all_onecol.csv")

    examples["train"] = [InputExample(f"train-{i}", row["Tweet"], None, row["Stance 1"]) for i, row in enumerate (train_examples.iloc)]
    examples["dev"] = [InputExample(f"dev-{i}", row["Tweet"], None, row["Stance 1"]) for i, row in enumerate (dev_examples.iloc)]
    examples["test"] = [InputExample(f"test-{i}", row["Tweet"], None, row["Stance 1"]) for i, row in enumerate (test_examples.iloc)]

    return examples, ["FAVOR", "AGAINST", "NONE"]


def get_data(task, train_num_per_class, dev_num_per_class, imbalance_rate,
             data_seed):
    if task in ['covid-19-stance']:
        return get_covid19stance_data()

    if task in ['sst-2', 'sst-5']:
        return get_text_data(
            task=task,
            train_num_per_class=train_num_per_class,
            dev_num_per_class=dev_num_per_class,
            imbalance_rate=imbalance_rate,
            data_seed=data_seed)

    elif task in ['cifar-10']:
        return get_image_data(
            train_num_per_class=train_num_per_class,
            dev_num_per_class=dev_num_per_class,
            imbalance_rate=imbalance_rate,
            data_seed=data_seed)