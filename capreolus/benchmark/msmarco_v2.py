import os
import json
from collections import defaultdict


from capreolus import constants, Dependency, constants
from capreolus.utils.loginit import get_logger

from . import Benchmark, IRDBenchmark

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class MSMARCOPassageV2(IRDBenchmark):
    """
    Qrels and training set data for MS MARCO Passage v2: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html#passage-ranking-dataset
    """

    module_name = "mspsg_v2"
    query_type = "text"
    ird_dataset_names = ["msmarco-passage-v2/train", "msmarco-passage-v2/dev1", "msmarco-passage-v2/trec-dl-2021"]
    dependencies = [Dependency(key="collection", module="collection", name="mspsg_v2")]
    fold_file = PACKAGE_PATH / "data" / "msmarcov2_passage_title_folds.json"
