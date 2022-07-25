from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
import unittest
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt


class SynapseDatasetTestCase(unittest.TestCase):
    def test_present(self):
        db_train = Synapse_dataset(base_dir="./data/Synapse/train_npz/", list_dir=os.path.join("lists", "lists_Synapse"),
                                   split="train",
                                   transform=transforms.Compose([]))
                                       # [RandomGenerator(output_size=[224, 224])]))
        if not os.path.exists("image"):
            os.mkdir("image")
        if not os.path.exists("image_label"):
            os.mkdir("image_label")
        for sample in db_train:
            if "slice070" not in sample["case_name"]:
                continue
            print(np.unique(sample["label"]))
            print(sample)
            # plt.imshow(sample["image"].squeeze())
            # plt.show()
            plt.imsave(os.path.join("image", sample["case_name"] + ".jpg"), np.rot90(sample["image"].squeeze(), k=3))
            # plt.imshow(sample["label"].squeeze())
            plt.imsave(os.path.join("image_label", sample["case_name"] + "_label.jpg"), np.rot90(sample["label"].squeeze(), k=3))

