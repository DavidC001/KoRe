import json
from typing import Tuple, Dict, Set

import networkx as nx
from datasets import Dataset
from tqdm import tqdm

from KG_LM.utils.Datasets.factories.WebQSPSentences import WebQSPSentences
from KG_LM.utils.Datasets.factories.WebQSPStar import WebQSPStar
from KG_LM.utils.Datasets.factories.TRExBite import TRExBite
from KG_LM.utils.Datasets.factories.TRExBiteLite import TRExBiteLite
from KG_LM.utils.Datasets.factories.TriREx import TriREx
from KG_LM.utils.Datasets.factories.TriRExLite import TriRExLite
from KG_LM.utils.Datasets.factories.TREx import TREx
from KG_LM.utils.Datasets.factories.TRExLite import TRExLite
from KG_LM.utils.Datasets.factories.TRExStar import TRExStar
from KG_LM.utils.Datasets.factories.TRExStarLite import TRExStarLite
from KG_LM.utils.Datasets.factories.GrailQA import GrailQA
from KG_LM.utils.Datasets.factories.GrailQAStar import GrailQAStar
from KG_LM.utils.Datasets.factories.SimpleQuestionsSentences import SimpleQuestionsSentences
from KG_LM.utils.Datasets.factories.SimpleQuestionsStar import SimpleQuestionsStar

from KG_LM.configuration import DatasetConfig

# Global cache for TRiREx training entities to avoid reloading
_TRIREX_TRAIN_ENTITIES_CACHE = None


def get_trirex_train_entities(conf: DatasetConfig) -> Set[str]:
    """
    Get the set of entity IDs from TRiREx training split.
    Uses caching to avoid reloading for multiple factory calls.
    """
    global _TRIREX_TRAIN_ENTITIES_CACHE
    
    if _TRIREX_TRAIN_ENTITIES_CACHE is not None:
        return _TRIREX_TRAIN_ENTITIES_CACHE
    
    print("Loading TRiREx train entities for filtering...")
    
    # Load TRiREx train split
    if conf.lite:
        trirex_builder = TriRExLite(conf.base_path)
    else:
        trirex_builder = TriREx(conf.base_path)
    
    if not trirex_builder.info.splits:
        trirex_builder.download_and_prepare()
    
    train_dataset = trirex_builder.as_dataset(split="train")
    
    # Extract entity IDs
    entity_ids = set()
    for sample in tqdm(train_dataset, desc="Extracting TRiREx train entities"):
        if 'subject' in sample and 'id' in sample['subject']:
            entity_ids.add(sample['subject']['id'])
    
    _TRIREX_TRAIN_ENTITIES_CACHE = entity_ids
    print(f"Cached {len(entity_ids)} TRiREx training entities for filtering")
    
    return entity_ids

def trex_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
    if conf.lite:
        trex_builder = TRExLite()
    else:
        trex_builder = TREx()

    if not trex_builder.info.splits:
        trex_builder.download_and_prepare()

    train_dataset = trex_builder.as_dataset(split="train")

    validation_dataset = trex_builder.as_dataset(split="validation")

    test_dataset = trex_builder.as_dataset(split="test")
    return train_dataset, validation_dataset, test_dataset


def trex_star_factory(conf: DatasetConfig) -> Dataset:
    if conf.lite:
        trex_star_builder = TRExStarLite(conf.base_path)
    else:
        trex_star_builder = TRExStar(conf.base_path)

    if not trex_star_builder.info.splits:
        trex_star_builder.download_and_prepare()

    return trex_star_builder.as_dataset(split="all")

def trex_star_graphs_factory(conf: DatasetConfig) -> Dict[str, nx.DiGraph]:
    dataset = trex_star_factory(conf)
    graphs = {}
    for datapoint in tqdm(dataset, desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)
    return graphs


def trex_bite_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
    if conf.lite:
        trex_bite_builder = TRExBiteLite(conf.base_path)
    else:
        trex_bite_builder = TRExBite(conf.base_path)

    if not trex_bite_builder.info.splits:
        trex_bite_builder.download_and_prepare()

    train_dataset = trex_bite_builder.as_dataset(split="train")
    validation_dataset = trex_bite_builder.as_dataset(split="validation")
    test_dataset = trex_bite_builder.as_dataset(split="test")
    return train_dataset, validation_dataset, test_dataset


def trirex_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
    if conf.lite:
        trirex_builder = TriRExLite(conf.base_path)
    else:
        trirex_builder = TriREx(conf.base_path)

    if not trirex_builder.info.splits:
        trirex_builder.download_and_prepare()

    train_dataset = trirex_builder.as_dataset(split="train")
    validation_dataset = trirex_builder.as_dataset(split="validation")
    test_dataset = trirex_builder.as_dataset(split="test")

    return train_dataset, validation_dataset, test_dataset


def web_qsp_factory(conf: DatasetConfig) -> Tuple[Dataset, Dict[str, nx.DiGraph]]:
    web_qsp_sentence_builder = WebQSPSentences(conf.base_path)
    web_qsp_star_builder = WebQSPStar(conf.base_path)

    if not web_qsp_sentence_builder.info.splits:
        web_qsp_sentence_builder.download_and_prepare()

    if not web_qsp_star_builder.info.splits:
        web_qsp_star_builder.download_and_prepare()

    test_dataset = web_qsp_sentence_builder.as_dataset(split="test")

    graphs = {}
    for datapoint in tqdm(web_qsp_star_builder.as_dataset(split="all"), desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    return test_dataset, graphs


def grailqa_factory(conf: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset, Dict[str, nx.DiGraph]]:
    """Factory function for GrailQA dataset with sentences and graphs."""
    grailqa_builder = GrailQA(conf.base_path)
    grailqa_star_builder = GrailQAStar(conf.base_path)

    if not grailqa_builder.info.splits:
        grailqa_builder.download_and_prepare()

    if not grailqa_star_builder.info.splits:
        grailqa_star_builder.download_and_prepare()

    train_dataset = grailqa_builder.as_dataset(split="train")
    validation_dataset = grailqa_builder.as_dataset(split="validation")
    test_dataset = grailqa_builder.as_dataset(split="test")

    # Load graphs
    graphs = {}
    for datapoint in tqdm(grailqa_star_builder.as_dataset(split="all"), desc="Loading GrailQA nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    # filter out all samples from dataset which do not have a graph
    train_dataset = train_dataset.filter(lambda x: x['subject']['id'] in graphs)
    validation_dataset = validation_dataset.filter(lambda x: x['subject']['id'] in graphs)
    test_dataset = test_dataset.filter(lambda x: x['subject']['id'] in graphs)

    return (train_dataset, validation_dataset, test_dataset), graphs


def simplequestions_factory(conf: DatasetConfig) -> Tuple[Tuple[Dataset, Dataset, Dataset], Dict[str, nx.DiGraph]]:
    """Factory function for SimpleQuestions dataset with sentences and graphs."""
    simplequestions_builder = SimpleQuestionsSentences(conf.base_path)
    simplequestions_star_builder = SimpleQuestionsStar(conf.base_path)

    if not simplequestions_builder.info.splits:
        simplequestions_builder.download_and_prepare()

    if not simplequestions_star_builder.info.splits:
        simplequestions_star_builder.download_and_prepare()

    train_dataset = simplequestions_builder.as_dataset(split="train")
    validation_dataset = simplequestions_builder.as_dataset(split="validation")
    test_dataset = simplequestions_builder.as_dataset(split="test")

    # Load graphs
    graphs = {}
    for datapoint in tqdm(simplequestions_star_builder.as_dataset(split="all"), desc="Loading SimpleQuestions nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    # filter out all samples from dataset which do not have a graph
    train_dataset = train_dataset.filter(lambda x: x['subject']['id'] in graphs)
    validation_dataset = validation_dataset.filter(lambda x: x['subject']['id'] in graphs)
    test_dataset = test_dataset.filter(lambda x: x['subject']['id'] in graphs)

    return (train_dataset, validation_dataset, test_dataset), graphs
