from __future__ import annotations

import os
import shutil
import warnings
import zipfile
import json
import yaml

from typing_extensions import override

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch

from torch import nn

import tensorflow as tf

from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
from pathlib import Path

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
from matgl.models import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")


class MLM(ModelLightningModule):
    @override
    def loss_fn(
        self,
        loss: nn.Module,
        labels: torch.Tensor,
        preds: torch.Tensor,
    ):
        """Args:
            loss: Loss function.
            labels: Labels to compute the loss.
            preds: Predictions.

        Returns:
            {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse, "EwT": ewt}
        """
        scaled_pred = torch.reshape(preds * self.data_std + self.data_mean, labels.size())
        total_loss = loss(labels, scaled_pred)
        mae = self.mae(labels, scaled_pred)
        rmse = self.rmse(labels, scaled_pred)

        e_thresh = 0.02
        error_energy = torch.abs(labels - preds)
        success = torch.count_nonzero(error_energy < e_thresh)
        total = labels.size(dim=0)
        # ewt = success / tf.cast(total, tf.int64)
        ewt = success / total

        return {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse, "EwT": ewt}


def cation_vacancy(
    pymatgen_dict: Structure,
    coord_a: float = 0.041667,
    coord_b: float = 0.083333,
):
    vacancy_coords_list = []
    for i in range(8):
        for j in range(8):
            vacancy_coords_list.append([coord_a + 0.125 * i,
                                        coord_b + 0.125 * j,
                                        0.25])
    for i in pymatgen_dict:
        coords = [round(float(i.a), 6),
                  round(float(i.b), 6),
                  round(float(i.c), 6)]
        if coords in vacancy_coords_list:
            vacancy_coords_list.remove(coords)

    for i in range(len(vacancy_coords_list)):
        pymatgen_dict.append("Cr", vacancy_coords_list[i], False)

    return pymatgen_dict


def anion_vacancy(
    pymatgen_dict: Structure,
    coord_a: float = 0.083333,
    coord_b: float = 0.041667,
    coord_c: float = 0.144826,
    first_second_layer_distance: float = 0.210348,
):
    vacancy_coords_list = []
    coords_list_1_layer = []
    coords_list_2_layer = []
    # 1st anion layer
    for i in range(8):
        for j in range(8):
            coords_list_1_layer.append([coord_a + 0.125 * i,
                                        coord_b + 0.125 * j,
                                        coord_c])
    # 2nd anion layer
    for i in range(8):
        for j in range(8):
            coords_list_2_layer.append([coord_a + 0.125 * i,
                                        coord_b + 0.125 * j,
                                        coord_c + first_second_layer_distance])
    # 1st anion layer
    for i in pymatgen_dict:
        coords = [round(float(i.a), 6), round(float(i.b), 6), round(float(i.c), 6)]
        if coords in coords_list_1_layer:
            coords_list_1_layer.remove(coords)

    # 2nd anion layer
    for i in pymatgen_dict:
        coords = [round(float(i.a), 6), round(float(i.b), 6), round(float(i.c), 6)]
        if coords in coords_list_2_layer:
            coords_list_2_layer.remove(coords)

    # 1st anion layer
    for i in range(len(coords_list_1_layer)):
        pymatgen_dict.append("O", coords_list_1_layer[i], False)

    # 2nd anion layer
    for i in range(len(coords_list_2_layer)):
        pymatgen_dict.append("O", coords_list_2_layer[i], False)

    return pymatgen_dict


def data_preprocessing(pymatgen_dict: Structure, cation: int = 0, anion: int = 0):
    formula = str(pymatgen_dict.formula).split(" ")
    for i in formula:
        if "Mo" in i or "W" in i:
            cation += int(i.lstrip("MoW"))
        if "S" in i or "Se" in i:
            anion += int(i.lstrip("SeS"))
    if cation < 64:
        pymatgen_dict = cation_vacancy(pymatgen_dict)
    if anion < 128:
        pymatgen_dict = anion_vacancy(pymatgen_dict)
    return pymatgen_dict


def read_pymatgen_dict(file, encoding="utf-8"):
    with open(file, "r", encoding=encoding) as f:
        d = json.load(f)
    return data_preprocessing(Structure.from_dict(d))


def energy_within_threshold(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.

    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)


def prepare_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)

    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }

    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values(), targets=targets)

    indexes = data.index.values
    structures = data.structures.values
    targets = data.targets.values

    return indexes, structures, targets


def main():
    CONFIG_PATH = Path.cwd() / "dataset" / "config.yaml"

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    indexes, structures, targets = prepare_dataset(config["datapath"])

    # get element types in the dataset
    elem_list = get_element_list(structures)

    # setup a graph converter
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)

    # convert the raw dataset into MEGNetDataset
    mp_dataset = MGLDataset(
        structures=list(structures),
        labels={"band_gap": targets},
        converter=converter,
    )

    # We will then split the dataset into training, validation and test data.
    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=666,
    )

    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn,
        batch_size=2,
        num_workers=0,
    )

    # setup the embedding layer for node attributes
    node_embed = torch.nn.Embedding(len(elem_list), 16)

    # define the bond expansion
    bond_expansion = BondExpansion(
        rbf_type="Gaussian",
        initial=0.0,
        final=5.0,
        num_centers=100,
        width=0.5,
    )

    # setup the architecture of MEGNet model
    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 64, 32),
        nlayers_set2set=1,
        niters_set2set=2,
        hidden_layer_sizes_output=(32, 16),
        is_classification=False,
        activation_type="softplus2",
        bond_expansion=bond_expansion,
        cutoff=4.0,
        gauss_width=0.5,
    )

    # setup the MEGNetTrainer
    lit_module = MLM(model=model)

    logger = CSVLogger(
        "logs",
        name="MEGNet_training",
    )
    trainer = pl.Trainer(max_epochs=20, accelerator="cpu", logger=logger)
    trainer.fit(
        model=lit_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
