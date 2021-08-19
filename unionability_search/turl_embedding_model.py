# Code is inspired by https://github.com/sunlab-osu/TURL/blob/release_ongoing/evaluate_task.ipynb (Access 08-13-2021)

from __future__ import absolute_import, division, print_function

import numpy as np
import sys

TURL_PATH = '../TURL/'
sys.path.insert(0, TURL_PATH)


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import trange
from tqdm.autonotebook import tqdm

from data_loader.hybrid_data_loaders import *
from data_loader.header_data_loaders import *
from data_loader.CT_Wiki_data_loaders import *
from data_loader.RE_data_loaders import *
from data_loader.EL_data_loaders import *
from model.configuration import TableConfig
from model.model import HybridTableMaskedLM, HybridTableCER, TableHeaderRanking, HybridTableCT, HybridTableEL, HybridTableRE, BertRE
from model.transformers import BertConfig, BertTokenizer, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils.util import *
from baselines.row_population.metric import average_precision, ndcg_at_k
from baselines.cell_filling.cell_filling import *
from model import metric

from statistics import mean_vector_similarity

logger = logging.getLogger(__name__)

BERT_TOKENIZER_PATH = 'bert-base-uncased'

MODEL_CLASSES = {
    'CER': (TableConfig, HybridTableCER, BertTokenizer),
    'CF': (TableConfig, HybridTableMaskedLM, BertTokenizer),
    'HR': (TableConfig, TableHeaderRanking, BertTokenizer),
    'CT': (TableConfig, HybridTableCT, BertTokenizer),
    'EL': (TableConfig, HybridTableEL, BertTokenizer),
    'RE': (TableConfig, HybridTableRE, BertTokenizer),
    'REBERT': (BertConfig, BertRE, BertTokenizer)
}

ENT_MASK_ID = 1


EPSILON = 1e-10  # only to prevent division by zero


class TurlEmbeddingModel:
    def __init__(self, model_name=(TURL_PATH + 'data/')):
        config_name = TURL_PATH + 'configs/table-base-config_v2.json'
        self.device = torch.device('cpu')

        config_class, model_class, _ = MODEL_CLASSES['CF']
        config = config_class.from_pretrained(config_name)
        config.output_attentions = True
        checkpoint = model_name
        self.model = model_class(config, is_simple=True)
        checkpoint = torch.load(os.path.join(
            checkpoint, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH)

    def build_input(self, pgEnt, pgTitle, secTitle, caption, headers, core_entities, core_entities_text, entity_cand, max_title_length, max_header_length, max_cell_length):
        tokenized_pgTitle = self.tokenizer.encode(
            pgTitle, max_length=max_title_length, add_special_tokens=False)
        tokenized_meta = tokenized_pgTitle +\
            self.tokenizer.encode(
                secTitle, max_length=max_title_length, add_special_tokens=False)
        if caption != secTitle:
            tokenized_meta += self.tokenizer.encode(
                caption, max_length=max_title_length, add_special_tokens=False)

        tokenized_headers = [self.tokenizer.encode(
            header, max_length=max_header_length, add_special_tokens=False) for header in headers]
        input_tok = []
        input_tok_pos = []
        input_tok_type = []
        tokenized_meta_length = len(tokenized_meta)
        input_tok += tokenized_meta
        input_tok_pos += list(range(tokenized_meta_length))
        input_tok_type += [0] * tokenized_meta_length
        header_span = []
        for tokenized_header in tokenized_headers:
            tokenized_header_length = len(tokenized_header)
            header_span.append([len(input_tok), len(
                input_tok) + tokenized_header_length])
            input_tok += tokenized_header
            input_tok_pos += list(range(tokenized_header_length))
            input_tok_type += [1] * tokenized_header_length

        input_ent = [0]
        input_ent_text = [tokenized_pgTitle[:max_cell_length]]
        input_ent_type = [2]

        # core entities in the subject column
        input_ent += [entity for entity in core_entities]
        input_ent_text += [self.tokenizer.encode(entity_text, max_length=max_cell_length, add_special_tokens=False) if len(
            entity_text) != 0 else [] for entity_text in core_entities_text]
        input_ent_type += [3] * len(core_entities)

        input_ent_cell_length = [len(x) if len(
            x) != 0 else 1 for x in input_ent_text]
        max_cell_length = max(input_ent_cell_length)
        input_ent_text_padded = np.zeros(
            [len(input_ent_text), max_cell_length], dtype=int)
        for i, x in enumerate(input_ent_text):
            input_ent_text_padded[i, :len(x)] = x
        assert len(input_ent) == 1 + 1 * len(core_entities)
        input_tok_mask = np.ones(
            [1, len(input_tok), len(input_tok) + len(input_ent)], dtype=int)
        input_tok_mask[0, header_span[0][0]:header_span[0]
                       [1], len(input_tok) + 1 + len(core_entities):] = 0
        input_tok_mask[0, :, len(input_tok) + 1 + len(core_entities):] = 0

        # build the mask for entities
        input_ent_mask = np.ones(
            [1, len(input_ent), len(input_tok) + len(input_ent)], dtype=int)
        input_ent_mask[0, 1 + len(core_entities):,
                       header_span[0][0]:header_span[0][1]] = 0

        input_tok_mask = torch.LongTensor(input_tok_mask)
        input_ent_mask = torch.LongTensor(input_ent_mask)

        input_tok = torch.LongTensor([input_tok])
        input_tok_type = torch.LongTensor([input_tok_type])
        input_tok_pos = torch.LongTensor([input_tok_pos])

        input_ent = torch.LongTensor([input_ent])
        input_ent_text = torch.LongTensor([input_ent_text_padded])
        input_ent_cell_length = torch.LongTensor([input_ent_cell_length])
        input_ent_type = torch.LongTensor([input_ent_type])

        input_ent_mask_type = torch.zeros_like(input_ent)

        candidate_entity_set = []
        candidate_entity_set = torch.LongTensor([candidate_entity_set])
        return input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent, input_ent_text, input_ent_cell_length, input_ent_type, input_ent_mask_type, input_ent_mask, candidate_entity_set

    def get_vectors(self, text_values, header='', norm=False):
        core_entities = list(range(10, len(text_values) + 10))
        core_entities_text = text_values
        input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, candidate_entity_set = self.build_input(
            -1, '', '', '', [header], core_entities, core_entities_text, None, 0, 20, 20)
        input_tok = input_tok.to(self.device)
        input_tok_type = input_tok_type.to(self.device)
        input_tok_pos = input_tok_pos.to(self.device)
        input_tok_mask = input_tok_mask.to(self.device)
        input_ent_text = input_ent_text.to(self.device)
        input_ent_text_length = input_ent_text_length.to(self.device)
        input_ent = input_ent.to(self.device)
        input_ent_type = input_ent_type.to(self.device)
        input_ent_mask_type = input_ent_mask_type.to(self.device)
        input_ent_mask = input_ent_mask.to(self.device)
        candidate_entity_set = candidate_entity_set.to(self.device)
        with torch.no_grad():
            tok_outputs, ent_outputs = self.model(input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent_text,
                                                  input_ent_text_length, input_ent_mask_type, input_ent, input_ent_type, input_ent_mask, candidate_entity_set)
            data_embeddings = ent_outputs[1]
            header_embeddings = tok_outputs[1]
            print('entity_text', text_values[0])
            print('data_embeddings.shape', data_embeddings.shape,
                  'len(text_values)', len(text_values))
            vectors = np.array(data_embeddings[0, :, :])
            if norm:
                vectors = [x / (np.linalg.norm(x) + EPSILON) for x in vectors]
            return vectors

    def get_header_vectors(self, text_values, header='', norm=False):
        core_entities = list(range(10, len(text_values) + 10))
        core_entities_text = text_values
        input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, candidate_entity_set = self.build_input(
            -1, '', '', '', [header], core_entities, core_entities_text, None, 0, 20, 20)
        input_tok = input_tok.to(self.device)
        input_tok_type = input_tok_type.to(self.device)
        input_tok_pos = input_tok_pos.to(self.device)
        input_tok_mask = input_tok_mask.to(self.device)
        input_ent_text = input_ent_text.to(self.device)
        input_ent_text_length = input_ent_text_length.to(self.device)
        input_ent = input_ent.to(self.device)
        input_ent_type = input_ent_type.to(self.device)
        input_ent_mask_type = input_ent_mask_type.to(self.device)
        input_ent_mask = input_ent_mask.to(self.device)
        candidate_entity_set = candidate_entity_set.to(self.device)
        with torch.no_grad():
            tok_outputs, ent_outputs = self.model(input_tok, input_tok_type, input_tok_pos, input_tok_mask, input_ent_text,
                                                  input_ent_text_length, input_ent_mask_type, input_ent, input_ent_type, input_ent_mask, candidate_entity_set)
            data_embeddings = ent_outputs[1]
            header_embeddings = tok_outputs[1]
            print('header_embeddings.shape', header_embeddings.shape,
                  'len(header)', len(header))
            vectors = np.array(header_embeddings[0, :, :])
            if norm:
                vectors = [x / (np.linalg.norm(x) + EPSILON) for x in vectors]
            return vectors

    def get_approximated_unionability_score(self, col1, col2, header1, header2,
                                            model_headers=True):
        score1, score2, score3 = None, None, None
        if model_headers:
            a_h = self.get_header_vectors([], header=header1)
            b_h = self.get_header_vectors([], header=header2)
            a = self.get_vectors(col1)
            b = self.get_vectors(col2)
            score1 = mean_vector_similarity(np.array(a), np.array(b))
            score2 = mean_vector_similarity(np.array(a), np.array(b_h))
            score3 = mean_vector_similarity(np.array(a_h), np.array(b))
        else:
            a = self.get_vectors(col1)
            b = self.get_vectors(col2)
            score1 = mean_vector_similarity(np.array(a), np.array(b))
            score2 = score1
            score3 = score1
        return (score1, score2, score3)
