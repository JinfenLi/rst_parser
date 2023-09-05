"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import os
import logging

from sgnlp.models.rst_pointer import RstPointerSegmenterConfig, RstPointerSegmenterModel, RstPreprocessor
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EDUSegmenter(object):

    def __init__(self, cache_dir):

        segment_path_to_exist = os.path.join(cache_dir, "segmenter")
        os.makedirs(segment_path_to_exist, exist_ok=True)
        if not os.path.exists(os.path.join(segment_path_to_exist, "config.json")):
            segmenter_config = RstPointerSegmenterConfig.from_pretrained(
                'https://storage.googleapis.com/sgnlp-models/models/rst_pointer/segmenter/config.json',
                cache_dir=segment_path_to_exist)
            segmenter_config.save_pretrained(segment_path_to_exist)
        else:
            logger.info(f"Loading EDU Segmenter config from {segment_path_to_exist}")
            segmenter_config = RstPointerSegmenterConfig.from_pretrained(
                os.path.join(segment_path_to_exist, "config.json"))
        if not os.path.exists(os.path.join(segment_path_to_exist, "pytorch_model.bin")):

            segmenter = RstPointerSegmenterModel.from_pretrained(
                'https://storage.googleapis.com/sgnlp-models/models/rst_pointer/segmenter/pytorch_model.bin',
                config=segmenter_config, cache_dir=segment_path_to_exist)

            segmenter.save_pretrained(segment_path_to_exist)
        else:
            logger.info(f"Loading EDU Segmenter model from {segment_path_to_exist}")
            segmenter = RstPointerSegmenterModel.from_pretrained(
                segment_path_to_exist, config=segmenter_config)
        self.segmenter = segmenter
        self.edu_preprocessor = RstPreprocessor()


    def __call__(self, texts):
        logger.info(f"Segmenting text ...")

        tokenized_sentences_ids, tokenized_sentences, lengths = self.edu_preprocessor(texts)
        segmenter_output = self.segmenter(tokenized_sentences_ids, lengths)
        end_boundaries = segmenter_output.end_boundaries
        return tokenized_sentences, end_boundaries



