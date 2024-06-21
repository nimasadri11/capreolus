import json

from capreolus import constants
from capreolus.utils.loginit import get_logger

from . import Collection, IRDCollection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class MSMARCO_PSG_V2(IRDCollection):
    """
    Collection for MS MARCO Passage v2: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html#passage-ranking-dataset
    """

    module_name = "mspsg_v2"
    ird_dataset_name = "msmarco-passage-v2"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        return json.dumps({"id": doc.doc_id, "contents": doc.text})

    def get_doc(self, docid):
        return self.docs_store.get(docid).text
