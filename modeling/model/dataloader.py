import copy
import csv
import json
import logging
import os
import sys

import pandas as pd
from torch.utils.data import (TensorDataset)

from utils import *

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, author_belief=None, author_belief_hl=None,
                 suspect_committedCrime=None, suspect_committedCrime_hl=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.author_belief = author_belief
        self.author_belief_hl = author_belief_hl
        self.suspect_committedCrime = suspect_committedCrime
        self.suspect_committedCrime_hl = suspect_committedCrime_hl

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label, highlight):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.highlight = highlight

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines


class GuiltProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, tokenizer=None, training_head=0, token_source=0):
        self.tokenizer = tokenizer
        self.training_head = training_head
        self.token_source = token_source
        assert isinstance(self.training_head, int)
        logger.info(f'GuiltProcessor training_head: {self.training_head}')

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "train.jsonl"), lines=True)
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "dev.jsonl"), lines=True)
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "test.jsonl"), lines=True)
        )

    def get_labels(self):
        """See base class."""
        return [0]

    def _create_examples(self, df: pd.DataFrame) -> list[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for rec in df.to_dict("records"):
            example = InputExample(
                guid=str(rec["story_id"]),
                text_a=rec["story_clean"],
                text_b=None,
                author_belief=rec["author_belief"],
                author_belief_hl=rec["author_belief_highlight"],
                suspect_committedCrime=rec["suspect_committedCrime"],
                suspect_committedCrime_hl=rec["suspect_committedCrime_highlight"]
            )
            if self.training_head == 0 and not np.isnan(example.author_belief):
                examples.append(example)
            elif self.training_head == 1 and not np.isnan(example.suspect_committedCrime):
                examples.append(example)
        logger.info(f'Created {len(examples)}/{len(df)} examples')
        return examples


def guilt_convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        highlight_pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    assert max_length == 400
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, truncation=True)

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        author_belief_hl = [0] + example.author_belief_hl[:max_length - 2] + [
            0] if example.author_belief_hl is not None else [0] * len(input_ids)
        suspect_committedCrime_hl = [0] + example.suspect_committedCrime_hl[:max_length - 2] + [
            0] if example.suspect_committedCrime_hl is not None else [0] * len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        # TODO refactor
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            if author_belief_hl:
                author_belief_hl = ([highlight_pad_token] * padding_length) + author_belief_hl
            if suspect_committedCrime_hl:
                suspect_committedCrime_hl = ([highlight_pad_token] * padding_length) + suspect_committedCrime_hl
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            author_belief_hl = author_belief_hl + ([highlight_pad_token] * padding_length)
            suspect_committedCrime_hl = suspect_committedCrime_hl + ([highlight_pad_token] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        if args.training_head[0] == 0:
            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    label=[example.author_belief], highlight=[author_belief_hl]
                )
            )
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    label=[example.suspect_committedCrime], highlight=[suspect_committedCrime_hl]
                )
            )

    return features


def load_and_cache_examples(args, task, tokenizer, split):
    logger.info(f'load and cache: {split}')
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    assert len(args.training_head) == 1
    processor = GuiltProcessor(tokenizer=tokenizer, training_head=args.training_head[0], token_source=args.token_source)
    output_mode = 'regression'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and False:  # disable loading cached features
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if split == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif split == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        elif split == 'test':
            examples = processor.get_test_examples(args.data_dir)
        else:
            assert f"no split found for {split}"
        features = guilt_convert_examples_to_features(args, examples,
                                                      tokenizer,
                                                      label_list=label_list,
                                                      max_length=args.max_seq_length,
                                                      output_mode=output_mode,
                                                      pad_on_left=bool(args.model_type in ['xlnet']),
                                                      # pad on the left for xlnet
                                                      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                          0],
                                                      pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                      )
        if args.local_rank in [-1, 0] and False:  # disable saving cached features
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    all_highlights = torch.tensor([f.highlight for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_highlights)
    return dataset
