"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import collections
import io
import logging
import os
import tarfile
import torch
import yaml
from lightning.pytorch import Trainer
from transformers import AutoTokenizer
from .data.data import DataModule
from .model.rst_model import RSTModel
from .utils.rst_tree.edu_segmenter import EDUSegmenter
from .utils.rst_tree.processor import RSTPreprocessor, RSTPostprocessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
cache_dir = os.path.join(os.path.dirname(__file__), f"model_dependencies")

def get_configuration():

    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(os.path.join(cache_dir, "rst-parser.tar.gz")):
        logger.info(f"Downloading model configuration files to {cache_dir} ...")
        torch.hub.download_url_to_file("https://pytorch-libs.s3.us-east-2.amazonaws.com/rst-parser.tar.gz",
                                       os.path.join(cache_dir, "rst-parser.tar.gz"))

        with tarfile.open(os.path.join(cache_dir, "rst-parser.tar.gz"), 'r:gz') as tar:
            tar.extractall(path=cache_dir)
    with open(os.path.join(cache_dir, "rst-parser", 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    ckpt_path = os.path.join(cache_dir, "rst-parser", 'pytorch_model.ckpt')
    return config, ckpt_path


def load_checkpoint(model, ckpt_path):

    buffer = io.BytesIO()
    torch.save(ckpt_path, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer)
    model = model.load_from_checkpoint(checkpoint, strict=False, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
    logger.info(f"Loaded checkpoint for evaluation from {ckpt_path}")
    return model

def parse(texts: list):
    """

    Args:
        texts: a list of texts

    Returns:
        a list of RST trees, a list of .dis files

    """

    config, ckpt_path = get_configuration()
    assert isinstance(texts, list), "input must be a list of texts"
    max_length = config['max_length']
    arch = config['arch']

    tokenizer = AutoTokenizer.from_pretrained(arch, strip_accents=False)
    rst_preprocessor = RSTPreprocessor(tokenizer, max_length)
    dataset_dict = collections.defaultdict(list)
    edu_segmenter = EDUSegmenter(cache_dir)
    tokenized_sentences, end_boundaries = edu_segmenter(texts)
    for i, doc in enumerate(texts):
        sentence = tokenized_sentences[i]
        boundaries = end_boundaries[i]
        cur_start = 0
        edus = []
        for b in boundaries:
            edus.append(" ".join(sentence[cur_start: b + 1]))
            cur_start = b + 1

        if len(edus) < 2:
            raise ValueError(f"Document {i} has less than 2 EDUs")
        features = collections.defaultdict(list)
        for edu in edus:
            input_ids, attention_mask = rst_preprocessor.process_edus(edu)
            features['edu_input_ids'].append(input_ids)
            features['edu_attention_masks'].append(attention_mask)
        dataset_dict['item_idx'].append(i)
        dataset_dict['edu_input_ids'].append(features['edu_input_ids'])
        dataset_dict['edu_attention_masks'].append(features['edu_attention_masks'])

    dm = DataModule()
    dm.setup(dataset=dataset_dict)
    loader = dm.predict_dataloader()

    model = RSTModel()
    model = load_checkpoint(model, ckpt_path)

    trainer = Trainer(logger=False)
    trainer.predict(model=model, dataloaders=loader)
    test_results = trainer.lightning_module.results
    dis_results = []
    tree_results = []
    for input_ids, prediction in zip(test_results['input_ids'], test_results['predictions']):
        rst_postprocessor = RSTPostprocessor(tokenizer)
        prediction = prediction.tolist()
        tree = rst_postprocessor.encode_tree(input_ids, prediction)
        tree_results.append(tree)
        rst_postprocessor.decode_tree(tree)
        dis_results.append(rst_postprocessor.dis_file)
        print(rst_postprocessor.dis_file)

    return tree_results, dis_results

