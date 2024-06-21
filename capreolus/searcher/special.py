import os
import gzip
import gdown
from pathlib import Path
from collections import defaultdict

from capreolus import ConfigOption, Dependency
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import download_file
from capreolus.utils.caching import cached_file, TargetFileExists

from . import Searcher
from .anserini import BM25

logger = get_logger(__name__)
import logging

logger.setLevel(logging.DEBUG)

SUPPORTED_TRIPLE_FILE = ["small", "large.v1", "large.v2"]


def _prepare_non_train_topic_fn(input_topics_fn, output_topics_fn, train_qids):
    """Extract the queries in `input_topics_fn` that not in training set into `output_topics_fn`"""
    if not isinstance(input_topics_fn, Path):
        input_topics_fn = Path(input_topics_fn)
    if not isinstance(output_topics_fn, Path):
        output_topics_fn = Path(output_topics_fn)

    if output_topics_fn.exists():
        logger.info(f"Use cached {output_topics_fn}.")
        return output_topics_fn

    output_topics_fn.parent.mkdir(parents=True, exist_ok=True)

    line_i = 0
    try:
        with cached_file(output_topics_fn) as tmp_fn:
            with open(input_topics_fn) as f, open(tmp_fn, "wt") as fout:
                for line in f:
                    line_i += 1
                    qid, _ = line.strip().split("\t")
                    if qid not in train_qids:
                        fout.write(line)
    except TargetFileExists as e:
        logger.info(f"Use cached file {output_topics_fn}.")
    except Exception as e:
        # note: file removal has been handled inside cached_file
        err_name = type(e).__name__
        line_info = f" (Line #{line_i})." if (line_i > 0) else ""
        logger.error(f"Encounter {err_name} while preparing non-train topic file: {input_topics_fn}{line_info}.")
        logging.error(f"Removing {output_topics_fn}.")
        raise e


class MsmarcoPsgSearcherMixin:
    # todo: avoid loading the entire runs into memory, combine two runfiles directly
    @staticmethod
    def convert_to_trec_runs(msmarco_top1k_fn, style="eval"):
        logger.info(f"Converting file {msmarco_top1k_fn} (with style {style}) into trec format")
        runs = defaultdict(dict)
        with open(msmarco_top1k_fn, "r", encoding="utf-8") as f:
            for line in f:
                if style == "triple":
                    qid, pos_pid, neg_pid = line.strip().split("\t")
                    runs[qid][pos_pid] = len(runs.get(qid, {}))
                    runs[qid][neg_pid] = len(runs.get(qid, {}))
                elif style == "eval":
                    qid, pid, _, _ = line.strip().split("\t")
                    runs[qid][pid] = len(runs.get(qid, []))
                else:
                    raise ValueError(f"Unexpected style {style}, should be either 'triple' or 'eval'")
        return runs

    @staticmethod
    def get_fn_from_url(url):
        return url.split("/")[-1].replace(".gz", "").replace(".tar", "")

    def get_url(self):
        tripleversion = self.config["tripleversion"]
        if tripleversion == "large.v1":
            return "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tsv.gz"

        if tripleversion == "large.v2":
            return "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz"

        if tripleversion == "small":
            return "https://drive.google.com/uc?id=1LCQ-85fx61_5gQgljyok8olf6GadZUeP"

        raise ValueError("Unknown version for triplet large" % self.config["tripleversion"])

    def download_and_prepare_train_set(self, tmp_dir):
        tmp_dir.mkdir(exist_ok=True, parents=True)
        triple_version = self.config["tripleversion"]

        url = self.get_url()
        if triple_version.startswith("large"):
            extract_file_name = self.get_fn_from_url(url)
            extract_dir = self.benchmark.collection.download_and_extract(url, tmp_dir, expected_fns=extract_file_name)
            triple_fn = extract_dir / extract_file_name
        elif triple_version == "small":
            triple_fn = tmp_dir / "triples.train.small.idversion.tsv"
            if not os.path.exists(triple_fn):
                gdown.download(url, triple_fn.as_posix(), quiet=False)
        else:
            raise ValueError(f"Unknown version for triplet: {triple_version}")

        return self.convert_to_trec_runs(triple_fn, style="triple")


class MSMARCO_V2_SearcherMixin:
    def get_train_runfile(self):
        raise NotImplementedError

    def combine_train_and_dev_runfile(self, dev_test_runfile, final_runfile, final_donefn):
        train_runfile = self.get_train_runfile()
        assert os.path.exists(dev_test_runfile)

        train_open_handler = gzip.open if train_runfile.suffix == ".gz" else open
        dev_open_handler = gzip.open if dev_test_runfile.suffix == ".gz" else open

        # write train and dev, test runs into final searcher file
        try:
            with cached_file(final_runfile) as tmp_fn:
                with open(tmp_fn, "w") as fout:
                    with train_open_handler(train_runfile) as fin:
                        for line in fin:
                            line = line if isinstance(line, str) else line.decode()
                            fout.write(line)

                    with dev_open_handler(dev_test_runfile) as fin:
                        for line in fin:
                            line = line if isinstance(line, str) else line.decode()
                            fout.write(line)

        except TargetFileExists as e:
            logger.info(f"Use cached file {final_runfile}.")
        except Exception as e:
            # note: file removal has been handled inside cached_file
            err_name = type(e).__name__
            logger.error(f"Encounter {err_name} while combining train and non-train runfile: {final_runfile}.")
            logging.error(f"Removing {final_runfile}.")
            raise e

        with open(final_donefn, "w") as f:
            f.write("done")


@Searcher.register
class MsmarcoPsg(Searcher, MsmarcoPsgSearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Use the offical runfile for the development and the test set.
    """

    module_name = "msmarcopsg"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]
    config_spec = [
        ConfigOption("tripleversion", "small", "version of triplet.qid file, small, large.v1 or large.v2"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        """only query results in dev and test set are saved"""
        final_runfn = output_path / "searcher"
        final_donefn = output_path / "done"
        if os.path.exists(final_donefn):
            return output_path

        tmp_dir = self.get_cache_path() / "tmp"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        output_path.mkdir(exist_ok=True, parents=True)

        # train
        train_run = self.download_and_prepare_train_set(tmp_dir=tmp_dir)
        self.write_trec_run(preds=train_run, outfn=final_runfn, mode="wt")

        # dev and test
        dev_test_urls = [
            "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
            "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz",
        ]
        runs = {}
        for url in dev_test_urls:
            extract_file_name = self.get_fn_from_url(url)
            extract_dir = self.benchmark.collection.download_and_extract(url, tmp_dir, expected_fns=extract_file_name)
            runs.update(self.convert_to_trec_runs(extract_dir / extract_file_name, style="eval"))
        self.write_trec_run(preds=runs, outfn=final_runfn, mode="a")

        with open(final_donefn, "wt") as f:
            print("done", file=f)
        return output_path


@Searcher.register
class MsmarcoPsgBm25(BM25, MsmarcoPsgSearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Conduct configurable BM25 search on the development and the test set.
    """

    module_name = "msmarcopsgbm25"
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="msmarcopsg"),
        Dependency(key="index", module="index", name="anserini"),
    ]
    config_spec = BM25.config_spec + [
        ConfigOption("tripleversion", "small", "version of triplet.qid file, small, large.v1 or large.v2"),
    ]

    def _query_from_file(self, topicsfn, output_path, config):
        final_runfn = os.path.join(output_path, "searcher")
        final_donefn = os.path.join(output_path, "done")
        if os.path.exists(final_donefn):
            return output_path

        output_path.mkdir(exist_ok=True, parents=True)
        tmp_dir = self.get_cache_path() / "tmp"
        tmp_topicsfn = tmp_dir / os.path.basename(topicsfn)
        tmp_output_dir = tmp_dir / "BM25_results"
        tmp_output_dir.mkdir(exist_ok=True, parents=True)

        train_runs = self.download_and_prepare_train_set(tmp_dir=tmp_dir)
        _prepare_non_train_topic_fn(
            input_topics_fn=topicsfn,
            output_topics_fn=tmp_topicsfn,
            train_qids=self.benchmark.folds["s1"]["train_qids"],
        )

        super()._query_from_file(topicsfn=tmp_topicsfn, output_path=tmp_output_dir, config=config)
        dev_test_runfile = tmp_output_dir / "searcher"
        assert os.path.exists(dev_test_runfile)

        # write train and dev, test runs into final searcher file
        Searcher.write_trec_run(train_runs, final_runfn)
        with open(dev_test_runfile) as fin, open(final_runfn, "a") as fout:
            for line in fin:
                fout.write(line)

        with open(final_donefn, "w") as f:
            f.write("done")
        return output_path


@Searcher.register
class MSMARCO_V2_Bm25(BM25, MSMARCO_V2_SearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Conduct configurable BM25 search on the development and the test set.
    """

    module_name = "msv2bm25"
    dependencies = [
        Dependency(key="benchmark", module="benchmark"),
        Dependency(key="index", module="index", name="anserini"),
    ]
    config_spec = BM25.config_spec

    def get_train_runfile(self):
        tmp_path = self.get_cache_path() / "tmp"
        if self.benchmark.module_name in ["mspsg_v2"]:
            url = "https://msmarco.blob.core.windows.net/msmarcoranking/passv2_train_top100.txt.gz"
            md5sum = "7cd731ed984fccb2396f11a284cea800"
        elif self.benchmark.module_name in ["msdoc_v2"]:
            url = "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_train_top100.txt.gz"
            md5sum = "b4d5915172d5f54bd23c31e966c114de"
        else:
            raise ValueError(
                f"Unexpected benchmark, should be either mspsg_v2 or msdoc_v2, but got {self.benchmark.module_name}."
            )

        gz_name = url.split("/")[-1]
        gz_file_path = tmp_path / gz_name
        download_file(url, gz_file_path, expected_hash=md5sum, hash_type="md5")
        return gz_file_path

    def _query_from_file(self, topicsfn, output_path, config):
        final_runfn = os.path.join(output_path, "searcher")
        final_donefn = os.path.join(output_path, "done")
        if os.path.exists(final_donefn):
            return output_path

        output_path.mkdir(exist_ok=True, parents=True)
        tmp_dir = self.get_cache_path() / "tmp"
        tmp_topicsfn = tmp_dir / os.path.basename(topicsfn)
        tmp_output_dir = tmp_dir / "BM25_results"
        tmp_output_dir.mkdir(exist_ok=True, parents=True)
        logger.debug("File output to - ", tmp_output_dir)

        # run bm25 on dev and test set
        _prepare_non_train_topic_fn(
            input_topics_fn=topicsfn,
            output_topics_fn=tmp_topicsfn,
            train_qids=self.benchmark.folds["s1"]["train_qids"],
        )

        logger.info("Searching non-training queries")
        super()._query_from_file(topicsfn=tmp_topicsfn, output_path=tmp_output_dir, config=config)
        self.combine_train_and_dev_runfile(tmp_output_dir / "searcher", final_runfn, final_donefn)
        return output_path


# todo: generalize the following two searchers
@Searcher.register
class MSMARCO_V2_Customize(Searcher, MSMARCO_V2_SearcherMixin):
    """This searcher allows to rerank an external runfile"""

    module_name = "msv2cust"
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="ms_v2"),
    ]
    config_spec = [
        ConfigOption("path", None, "path to the external trec-format runfile"),
    ]

    def get_train_runfile(self):
        basename = f"{self.benchmark.dataset_type}v2_train_top100.txt"
        return self.benchmark.data_dir / basename

    def _query_from_file(self, topicsfn, output_path, config):
        final_runfn = os.path.join(output_path, "searcher")
        final_donefn = os.path.join(output_path, "done")
        if os.path.exists(final_donefn):
            return output_path

        output_path.mkdir(exist_ok=True, parents=True)

        runfile_path = self.config["path"]
        if not os.path.exists(runfile_path):
            raise IOError(f"Could not find the provided runfile: {runfile_path}")

        self.combine_train_and_dev_runfile(runfile_path, final_runfn, final_donefn)
        return output_path


# todo: make this another type of "Module" (e.g. DPR Module)
@Searcher.register
class StaticTctColBertDev(Searcher, MsmarcoPsgSearcherMixin):
    """
    Skip the searching on training set by converting the official training triplet into a "fake" runfile.
    Use the runfile pre-prepared using TCT-ColBERT (https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf)
    """

    module_name = "static_tct_colbert"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]
    config_spec = [
        ConfigOption("tripleversion", "small", "version of triplet.qid file, small, large.v1 or large.v2"),
    ]

    def _query_from_file(self, topicsfn, output_path, cfg):
        outfn = output_path / "static.run"
        if outfn.exists():
            return outfn

        tmp_dir = self.get_cache_path() / "tmp"
        output_path.mkdir(exist_ok=True, parents=True)

        # train
        train_runs = self.download_and_prepare_train_set(tmp_dir=tmp_dir)
        self.write_trec_run(preds=train_runs, outfn=outfn, mode="wt")
        logger.info(f"prepared runs from train set")

        # dev
        tmp_dev = tmp_dir / "tct_colbert_v1_wo_neg.tsv"
        if not tmp_dev.exists():
            tmp_dir.mkdir(exist_ok=True, parents=True)
            url = "http://drive.google.com/uc?id=1jOVL3DIya6qDiwM_Dnqc81FT5ZB43csP"
            gdown.download(url, tmp_dev.as_posix(), quiet=False)

        assert tmp_dev.exists()
        with open(tmp_dev, "rt") as f, open(outfn, "at") as fout:
            for line in f:
                qid, docid, rank, score = line.strip().split("\t")
                fout.write(f"{qid} Q0 {docid} {rank} {score} tct_colbert\n")
        return outfn
