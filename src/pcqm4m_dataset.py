# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import datasets
import pyarrow as pa
import torch
from pyarrow import csv


class PCQM4MDataset(datasets.ArrowBasedBuilder):
    """SMILES dataset loader for OGB-LSC"""

    _CITATION = """"""
    _DESCRIPTION = ""
    _HOMEPAGE = ""
    _LICENSE = ""
    _URLs = {
        "pcqm4m_kddcup2021": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip"
    }
    VERSION = datasets.Version("1.1.1")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="pcqm4m_kddcup2021",
            version=VERSION,
            description="",
        ),
    ]
    DEFAULT_CONFIG_NAME = "pcqm4m_kddcup2021"

    def _info(self):
        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "smiles": datasets.Value("string"),
                    "homolumogap": datasets.Value("float32"),
                }
            ),
            supervised_keys=None,
            homepage=self._HOMEPAGE,
            license=self._LICENSE,
            citation=self._CITATION,
        )

    def _split_generators(self, dl_manager) -> [datasets.SplitGenerator]:
        urls = self._URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)

        splits = torch.load(os.path.join(data_dir, self.config.name, "split_dict.pt"))
        csv_file = os.path.join(data_dir, self.config.name, "raw", "data.csv.gz")

        return [
            datasets.SplitGenerator(
                name=datasets.Split(k),
                gen_kwargs={"file": csv_file, "index": v, "split": k},
            )
            for k, v in splits.items()
        ]

    def _generate_tables(self, file, index, split):
        data = csv.read_csv(file)
        df = data.to_pandas()
        yield split, pa.Table.from_pandas(df.loc[index])
