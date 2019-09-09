# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import json
import random
import numpy as np
import utils
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import preprocess

#os.environ['CUDA_VISIBLE_DEVICES'] = '5'


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 20,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("model_type", default='full', help="""full(default) | intent_only
                                                                    full: full attention model
                                                                    intent_only: intent attention model""")


flags.DEFINE_string('domain_vocab', default='domain_vocab.txt', help="domain vocab")
flags.DEFINE_string('intent_vocab', default='intent_vocab.txt', help="intent vocab")
flags.DEFINE_string('slot_vocab', default='slot_vocab.txt', help="slot_vocab")
flags.DEFINE_string('vocab_path', default='./', help='vocab')


flags.DEFINE_string("train_input_file", default='train_sentences.txt', help="Input file name.")
flags.DEFINE_string("train_slot_file",  default='train_role_labels.txt', help="Slot file name.")
flags.DEFINE_string("train_intent_file",  default='train_intents.txt', help="Intent file name.")
flags.DEFINE_string("train_domain_file",  default='train_domains.txt', help="domain file name")

flags.DEFINE_string("valid_input_file",  default='valid_sentences.txt', help="valid Input file name.")
flags.DEFINE_string("valid_slot_file",  default='valid_role_labels.txt', help="valid Slot file name.")
flags.DEFINE_string("valid_intent_file",  default='valid_intents.txt', help="valid Intent file name.")
flags.DEFINE_string("valid_domain_file", default='valid_domains.txt', help="valid domain file name")

flags.DEFINE_string("test_file",  default='sample.json', help="test_file")
flags.DEFINE_string("predict_file", default='predict.json', help="test_file")

flags.DEFINE_string("test_input_file",  default='test_sentences.txt', help="test Input file name.")
flags.DEFINE_string("test_slot_file",  default='test_role_labels.txt', help="test Slot file name.")
flags.DEFINE_string("test_intent_file",  default='test_intents.txt', help="test Intent file name.")
flags.DEFINE_string("test_domain_file",  default='test_domains.txt', help="test domain file name")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, domain_label=None, intent_label =None, slot_label = None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.domain_label = domain_label
        self.intent_label = str(intent_label)
        self.slot_label = slot_label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 input_ids,
                 input_mask,
                 segment_ids,
                 domain_label_id,
                 intent_label_id,
                 slot_label_id,
                 is_real_example=True):
        self.unique_id = unique_id
        self.example_index = example_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.domain_label_id = domain_label_id
        self.intent_label_id = intent_label_id
        self.slot_label_id = slot_label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DialogueProcessor(DataProcessor):
    """Processor for the smp task1 data set ."""

    def get_train_examples(self, data_dir):
        print(data_dir)
        """See base class."""
        return self._create_examples(
            open(os.path.join(data_dir, "sample.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            open(os.path.join(data_dir, "valid.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            open(os.path.join(data_dir, FLAGS.test_file)), "test")

    def get_domain_labels(self):
        """See base class."""
        domain_labels = []
        with open('domain_vocab') as reader:
            for line in reader.readlines():
                line = line.strip('\n')
                domain_labels.append(line)
        return domain_labels

    def get_intent_label(self):
        intent_labels = []
        with open('intent_vocab') as reader:
            for line in reader.readlines():
                line = line.strip('\n')
                intent_labels.append(line)
        return intent_labels

    def get_slot_label(self):
        slot_labels = []
        with open('slot_vocab') as reader:
            for line in reader.readlines():
                line = line.strip('\n')
                slot_labels.append(line)
        return slot_labels

    def get_vocab(self):
        vocab = {}
        with open('vocab.txt') as reader:
            for (i, line) in enumerate(reader.readlines()):
                line = line.strip('\n')
                vocab.update({i:line})
        return vocab

    def _create_examples(self, reader, set_type):
        """Creates examples for the training and dev sets."""
        data = json.load(reader)
        examples = []
        if set_type =='test':
            for index, entry in enumerate(data):
                guid = "%s-%s"%(set_type,index)
                text_a = tokenization.convert_to_unicode(entry['text'])
                domain = ""
                intent = ""
                slot = None
                examples.append(InputExample(guid =guid, text_a =text_a, text_b=None, domain_label=domain, intent_label=intent, slot_label=None))
        else:
            for index, entry in enumerate(data):
                guid = "%s-%s"%(set_type, index)
                text_a = tokenization.convert_to_unicode(entry['text'])
                domain = entry['domain']
                intent = entry['intent']
                slots = entry['slots']
                slot_role_label = tokenization.sentence_role(text_a, slots)
                slot_role_label = slot_role_label.split()
                examples.append(InputExample(guid =guid, text_a =text_a, text_b=None, domain_label=domain, intent_label=intent, slot_label=slot_role_label))
        return examples



def convert_single_example(ex_index, example, domain_label_list, intent_label_list, slot_label_list, max_seq_length,
                           tokenizer, set_type):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    #
    # if isinstance(example, PaddingInputExample):
    #     return InputFeatures(
    #         unique_ids=example.guid,
    #         input_ids=[0] * max_seq_length,
    #         input_mask=[0] * max_seq_length,
    #         segment_ids=[0] * max_seq_length,
    #         label_id=0,
    #         is_real_example=False)
    start = 1000000000

    domain_label_map = {}
    intent_label_map = {}
    slot_label_map = {}

    for (i, label) in enumerate(domain_label_list):
        domain_label_map[label] = i
    for (i, label) in enumerate(intent_label_list):
        intent_label_map[label] = i
    for (i, label) in enumerate(slot_label_list):
        slot_label_map[label] = i

    # def not_empty(s):
    #     return s and s.strip()
    tokens_a = [c.lower() for c in tokenization.convert_to_unicode(example.text_a)]
    # tokens_a = list(filter(not_empty, tokens_a))
    tokens_b = None
    if example.text_b:
        # tokens_b = tokenizer.tokenize(example.text_b)
        toker_b = [c.lower for c in tokenization.convert_to_unicode(example.text_b)]
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]



    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    if set_type == 'test':
        domain_label_id = 0
        intent_label_id = 0
        tokens = []
        segment_ids = []
        slot_label_id = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        slot_label_id.append(slot_label_map['o'])

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
            slot_label_id.append(slot_label_map['o'])

        tokens.append("[SEP]")
        segment_ids.append(0)
        slot_label_id.append(slot_label_map['o'])

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
                slot_label_id.append(slot_label_map[0])

            tokens.append("[SEP]")
            segment_ids.append(1)
            slot_label_id.append(slot_label_map['o'])
    else:
        domain_label_id = domain_label_map[example.domain_label]
        intent_label_id = intent_label_map[example.intent_label]

        slot_label_list = example.slot_label
        if len(example.slot_label) > max_seq_length - 2:
            slot_label_list = example.slot_label[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        slot_label_id = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        slot_label_id.append(slot_label_map['o'])

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        for slot_label in slot_label_list:
            slot_label_id.append(slot_label_map[slot_label])

        tokens.append("[SEP]")
        segment_ids.append(0)
        slot_label_id.append(slot_label_map['o'])

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)

            for slot_label in slot_label_list:
                slot_label_id.append(slot_label_map[slot_label])

            tokens.append("[SEP]")
            segment_ids.append(1)
            slot_label_id.append(slot_label_map['o'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        slot_label_id.append(slot_label_map['o'])

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slot_label_id) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.domain_label, domain_label_id))
        tf.logging.info("label: %s (id = %d)" % (example.intent_label, intent_label_id))
        tf.logging.info("label: {} (id = {})".format(example.slot_label, domain_label_id))

    feature = InputFeatures(
        unique_id=start + ex_index,
        example_index=ex_index,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        domain_label_id=domain_label_id,
        intent_label_id=intent_label_id,
        slot_label_id=slot_label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, domain_label_list, intent_label_list, slot_label_list, max_seq_length, tokenizer, output_file, set_type):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    all_features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, domain_label_list, intent_label_list, slot_label_list,
                                         max_seq_length, tokenizer,set_type)
        all_features.append(feature)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["domain_label_ids"] = create_int_feature([feature.domain_label_id])
        features["intent_label_ids"] = create_int_feature([feature.intent_label_id])
        features["slot_label_ids"] = create_int_feature(feature.slot_label_id)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    return all_features


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "domain_label_ids": tf.FixedLenFeature([], tf.int64),
        "intent_label_ids": tf.FixedLenFeature([], tf.int64),
        "slot_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, domain_label_ids, intent_label_ids,
                 slot_label_ids, domain_num_labels, intent_num_labels, slot_num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    print('start to create model')
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    #model type
    add_final_state_to_intent = True
    add_final_state_to_domain = True
    remove_slot_attn = False
    if FLAGS.model_type == 'full':
        add_final_state_to_intent = True
        add_final_state_to_domain = True
        remove_slot_attn = False
    elif FLAGS.model_type == 'intent_only':
        add_final_state_to_intent = True
        add_final_state_to_domain = True
        remove_slot_attn = True
    else:
        print('unknown model type!')
        exit(1)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    state_outputs = model.get_sequence_output()
    print('state_outputs shape: ', state_outputs.get_shape())
    final_state = model.get_pooled_output()
    print('final_state shape: ', final_state.get_shape())
    state_shape = state_outputs.get_shape()
    '''
    ## bilstm
    with tf.variable_scope("bilstm"):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(state_shape[2].value)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(state_shape[2].value)

        if is_training == True:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
                                                    output_keep_prob=0.5)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
                                                    output_keep_prob=0.5)
        used = tf.sign(tf.abs(input_ids))
        sequence_lengths = tf.reduce_sum(used, reduction_indices=1)
        state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, state_outputs,
                                                                     sequence_length= sequence_lengths, dtype=tf.float32)

        # final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
        # state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
        print('type', type(final_state[0][0]))
        print('shape', final_state[0][0].get_shape())
        print('type', type(final_state[1][0]))
        print('shape', final_state[1][0].get_shape())
        final_state = tf.add(final_state[0][0],final_state[1][0])
        state_outputs = tf.add(state_outputs[0], state_outputs[1])
        print('bilstm final_state', final_state.get_shape())
        print('bilstm state output', state_outputs.get_shape())
    '''
    with tf.variable_scope('attention'):
        slot_inputs = state_outputs
        if remove_slot_attn == False:
            with tf.variable_scope('slot_attn'):
                attn_size = state_shape[2].value
                origin_shape = tf.shape(state_outputs)
                hidden = tf.expand_dims(state_outputs, 1)
                print("hidden_shape", hidden.get_shape())
                hidden_conv = tf.expand_dims(state_outputs, 2)
                k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
                hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
                hidden_features = tf.reshape(hidden_features, origin_shape)
                hidden_features = tf.expand_dims(hidden_features, 1)
                v = tf.get_variable("AttnV", [attn_size])

                slot_inputs_shape = tf.shape(slot_inputs)
                slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])
                y = core_rnn_cell._linear(slot_inputs, attn_size, True)
                y = tf.reshape(y, slot_inputs_shape)
                y = tf.expand_dims(y, 2)
                s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])
                a = tf.nn.softmax(s)

                a = tf.expand_dims(a, -1)
                slot_d = tf.reduce_sum(a * hidden, [2])
        else:
            attn_size = state_shape[2].value
            slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])

        intent_input = final_state
        with tf.variable_scope('intent_attn'):
            attn_size = state_shape[2].value
            hidden = tf.expand_dims(state_outputs, 2)
            k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [attn_size])

            y = core_rnn_cell._linear(intent_input, attn_size, True)
            y = tf.reshape(y, [-1, 1, 1, attn_size])
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            a = tf.expand_dims(a, -1)
            a = tf.expand_dims(a, -1)
            d = tf.reduce_sum(a * hidden, [1, 2])
            if add_final_state_to_intent == True:
                intent_output = tf.concat([d, intent_input], 1)
            else:
                intent_output = d

        domain_input = final_state
        with tf.variable_scope('domain_attn'):
            attn_size = state_shape[2].value
            hidden = tf.expand_dims(state_outputs, 2)
            k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [attn_size])

            y = core_rnn_cell._linear(domain_input, attn_size, True)
            y = tf.reshape(y, [-1, 1, 1, attn_size])
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            a = tf.expand_dims(a, -1)
            a = tf.expand_dims(a, -1)
            d = tf.reduce_sum(a * hidden, [1, 2])
            if add_final_state_to_domain == True:
                domain_output = tf.concat([d, domain_input], 1)
            else:
                domain_output = d



        with tf.variable_scope('slot_gated'):
            intent_gate = core_rnn_cell._linear(intent_output, attn_size, True)
            intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[1].value])
            print('intent_gate', intent_gate.get_shape())
            v1 = tf.get_variable("gateV", [attn_size])
            if remove_slot_attn == False:
                slot_gate = v1 * tf.tanh(slot_d + intent_gate)
                print('v1 shape', v1.get_shape())
                print('slot_d shape', slot_d.get_shape())
                print('slot_gate', slot_gate.get_shape())
            else:
                slot_gate = v1 * tf.tanh(state_outputs + intent_gate)
            slot_gate = tf.reduce_sum(slot_gate, [2])
            slot_gate = tf.expand_dims(slot_gate, -1)
            print('slot_gate', slot_gate.get_shape())
            if remove_slot_attn == False:
                slot_gate = slot_d * slot_gate
                print('slot_gate', slot_gate.get_shape())
            else:
                slot_gate = state_outputs * slot_gate
            slot_gate = tf.reshape(slot_gate, [-1, attn_size])
            slot_output = tf.concat([slot_gate, slot_inputs], 1)


        # with tf.variable_scope('intent_gated'):
        #     domain_gate = core_rnn_cell._linear(domain_output, attn_size, True)
        #     v1 = tf.get_variable('gateV', [attn_size])
        #     print('d', d.get_shape())
        #     print('domain_gate', domain_gate.get_shape())
        #     intent_gate = v1 * tf.tanh(d + domain_gate)
        #     print('intent_gate', intent_gate.get_shape())
        #     intent_gate = d * intent_gate
        #     intent_gate = tf.reshape(intent_gate, [-1, attn_size])
        #     intent_output = tf.concat([intent_gate, intent_input], 1)

    with tf.variable_scope('intent_proj'):
        intent = core_rnn_cell._linear(intent_output, intent_num_labels, True)

    with tf.variable_scope('dimain_proj'):
        domain = core_rnn_cell._linear(domain_output, domain_num_labels, True)

    with tf.variable_scope('slot_proj'):
        slot = core_rnn_cell._linear(slot_output, slot_num_labels, True)
        slot = tf.reshape(slot, [-1, state_shape[1].value, slot_num_labels])
        used = tf.sign(tf.abs(input_ids))
        sequence_lengths = tf.reduce_sum(used, reduction_indices=1)
        per_slot_loss,trans = tf.contrib.crf.crf_log_likelihood(slot, slot_label_ids, sequence_lengths, transition_params=None)
        slot_predict,_= tf.contrib.crf.crf_decode(slot,trans,sequence_lengths)
        slot_loss = tf.reduce_sum(-per_slot_loss)

    print('slot label ids shape', slot_label_ids.get_shape())  #(batch, length)
    slot_label_ids = tf.reshape(slot_label_ids, [-1,1])
    slot_label_ids = tf.one_hot(slot_label_ids, slot_num_labels,axis=1)
    slot_label_ids = tf.squeeze(slot_label_ids)
    slot_predict = tf.reshape(slot_predict, [-1,1])
    slot_predict = tf.one_hot(slot_predict, slot_num_labels,axis=1)
    slot_predict = tf.squeeze(slot_predict) ##(batch*length, num_slot_label)
    print('slot label ids shape', slot_label_ids.get_shape())

    intent_label_ids = tf.one_hot(intent_label_ids, intent_num_labels,axis=1)
    print('intent_label_id shape', intent_label_ids.get_shape)
    print('intent shape', intent.get_shape)

    domain_label_ids = tf.one_hot(domain_label_ids, domain_num_labels, axis =1)
    print('domain_label_ids shape', domain_label_ids.get_shape)
    print('domain shape', domain)

    with tf.variable_scope('intent_loss'):
        intent_crossent = tf.nn.softmax_cross_entropy_with_logits(labels=intent_label_ids, logits=intent)
        # intent_label_shape = intent_label_ids.get_shape()
        print('intent_crossent shape', intent_crossent.get_shape())
        intent_loss = tf.reduce_sum(intent_crossent)

    with tf.variable_scope('domain_loss'):
        domain_crossent = tf.nn.softmax_cross_entropy_with_logits(labels=domain_label_ids, logits=domain)
        # domain_label_shape = domain_label_ids.get_shape()
        domain_loss = tf.reduce_sum(domain_crossent)

    with tf.variable_scope('slot_loss'):
        slot_crossent = tf.nn.softmax_cross_entropy_with_logits(labels=slot_label_ids, logits=slot_predict)
        # crossent = tf.reshape(crossent)
        # slot_loss = tf.reduce_sum(slot_crossent)

    slot_output = tf.nn.softmax(slot_predict, name='slot_output', axis=-1)
    intent_output = tf.nn.softmax(intent, name='intent_output')
    domain_output = tf.nn.softmax(domain, name='domain_output')

    total_loss = intent_loss + domain_loss + slot_loss

    return (slot_output, intent_output, domain_output, total_loss, -per_slot_loss, intent_crossent, domain_crossent)



def model_fn_builder(bert_config, domain_num_labels, intent_num_labels, slot_num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""
    print('start to build model_fn')

    def model_fn(features, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s , type = %s" % (name, features[name].shape, features[name].dtype))
        unique_ids = features['unique_ids']
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        domain_label_ids = features["domain_label_ids"]
        intent_label_ids = features["intent_label_ids"]
        slot_label_ids = features["slot_label_ids"]

        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(domain_num_labels), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        print("*****************************")
        (slot_output, intent_output, domain_output, total_loss, per_slot_loss, per_intent_loss, per_domain_loss) = create_model(bert_config,
           is_training, input_ids, input_mask, segment_ids, domain_label_ids, intent_label_ids, slot_label_ids,
            domain_num_labels, intent_num_labels, slot_num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_intent_loss, per_domain_loss, per_slot_loss, slot_output, intent_output, domain_output,
                          domain_label_ids, intent_label_ids, slot_label_ids):
                intent_predict = tf.argmax(intent_output, axis =-1, output_type=tf.int32)
                domain_predict = tf.argmax(domain_output, axis = -1, output_type=tf.int32)
                slot_output_predict = tf.argmax(slot_output, axis = -1, output_type= tf.int32)
                slot_output_predict = tf.reshape(slot_output_predict, [-1, FLAGS.max_seq_length])
                domain_accuracy = tf.metrics.accuracy(labels=domain_label_ids, predictions=domain_predict)
                intent_accuracy = tf.metrics.accuracy(labels=intent_label_ids, predictions=intent_predict)
                slot_accuracy = tf.metrics.accuracy(labels=slot_label_ids, predictions=slot_output_predict)
                slot_loss = tf.metrics.mean(per_slot_loss)
                domain_loss = tf.metrics.mean(per_domain_loss)
                intent_loss = tf.metrics.mean(per_intent_loss)
                total_loss = slot_loss + domain_loss + intent_loss

                return {
                    "domain_accuracy": domain_accuracy,
                    "intent_accuracy": intent_accuracy,
                    "slot_accuracy": slot_accuracy,
                    "eval_intent_loss": intent_loss,
                    "eval_domain_loss": domain_loss,
                    "eval__slot_loss": slot_loss,
                }

            eval_metrics = (metric_fn,
                            [per_intent_loss, per_domain_loss, per_slot_loss, slot_output, intent_output, domain_output, domain_label_ids, intent_label_ids, slot_label_ids])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:

            intent_predict = tf.argmax(intent_output, axis=-1, output_type=tf.int32)
            domain_predict = tf.argmax(domain_output, axis=-1, output_type=tf.int32)
            slot_output_predict = tf.argmax(slot_output, axis=-1, output_type=tf.int32)
            slot_output_predict = tf.reshape(slot_output_predict, [-1, FLAGS.max_seq_length])
            predict_result = {'intent_predict':intent_predict, 'domain_predict':domain_predict, 'slot_output_predict':slot_output_predict}

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predict_result,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    # all_label_ids = []
    all_domain_label_ids = []
    all_intent_label_ids = []
    all_slot_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_domain_label_ids.append(feature.domain_label_id)
        all_intent_label_ids.append(feature.intent_label_id)
        all_slot_label_ids.append(feature.slot_label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int64),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int64),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int64),
            "domain_label_ids":
                tf.constant(all_domain_label_ids, shape=[num_examples], dtype=tf.int64),
            "intent_label_ids":
                tf.constant(all_intent_label_ids, shape=[num_examples], dtype=tf.int64),
            "slot_label_ids":
                tf.constant(all_slot_label_ids, shape=[num_examples, seq_length], dtype= tf.int64)
        }
         )

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def write_predictions(vocab, predict_examples, predict_features, result, domain_label_list, intent_label_list, slot_label_list, num_actual_predict_examples, output_prediction_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" %output_prediction_file)
    all_predictions = []
    print(len(vocab))
    for i,prediction in enumerate(result):
        domain_id = prediction['domain_predict']
        intent_id = prediction['intent_predict']
        slot_id = prediction['slot_output_predict']
        slot_output = [slot_label_list[i] for i in slot_id]

        domain = domain_label_list[domain_id]
        intent = intent_label_list[intent_id]
        text = predict_examples[i].text_a
        input_ids = predict_features[i].input_ids
        input_word = [vocab[i] for i in input_ids]
        slots = utils.bioRestore(slot_output, input_word)
        all_predictions.append({'text':text, 'domain':domain, 'intent':intent, 'slots':slots})
        if i < 3:
            print(all_predictions[i])
            print('slot_id', slot_id)


    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    assert len(all_predictions) == num_actual_predict_examples



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "dialogue":DialogueProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    tokenization.createVocabularyChinese(os.path.join(FLAGS.data_dir, FLAGS.train_input_file),
                                          os.path.join(FLAGS.vocab_path, 'in_vocab'))
    tokenization.createVocabulary(os.path.join(FLAGS.data_dir, FLAGS.train_slot_file),
                                   os.path.join(FLAGS.vocab_path, 'slot_vocab'))
    tokenization.createVocabulary(os.path.join(FLAGS.data_dir, FLAGS.train_intent_file),
                                   os.path.join(FLAGS.vocab_path, 'intent_vocab'))
    tokenization.createVocabulary(os.path.join(FLAGS.data_dir, FLAGS.train_domain_file),
                             os.path.join(FLAGS.vocab_path, 'domain_vocab'))
    domain_label_list = processor.get_domain_labels()
    intent_label_list = processor.get_intent_label()
    slot_label_list = processor.get_slot_label()
    vocab = processor.get_vocab()


    tf.logging.info('number of domain %d', len(domain_label_list))
    tf.logging.info('number of intent %d', len(intent_label_list))
    tf.logging.info('number of slot %d', len(slot_label_list))


    if FLAGS.do_train:
        preprocess.trans_type('./sample.json', 'train')

        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        domain_num_labels=len(domain_label_list),
        intent_num_labels=len(intent_label_list),
        slot_num_labels=len(slot_label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    print('model_fn finished')


    # #
    # # # If TPU is not available, this will fall back to normal Estimator on CPU
    # # # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    print('estimator finished')

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, domain_label_list, intent_label_list, slot_label_list, FLAGS.max_seq_length, tokenizer, train_file, 'train')
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        tf.logging.info("  Num domain_label_ids%d", len(domain_label_list))
        tf.logging.info("  Num intent_label_ids%d", len(intent_label_list))
        tf.logging.info("  Num slot_label_ids%d", len(slot_label_list))

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    #

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        eval_features = file_based_convert_examples_to_features(
            eval_examples, domain_label_list, intent_label_list, slot_label_list, FLAGS.max_seq_length, tokenizer, eval_file, 'eval')

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        tf.logging.info("  Num domain_label_ids%d", len(domain_label_list))
        tf.logging.info("  Num intent_label_ids%d", len(intent_label_list))
        tf.logging.info("  Num slot_label_ids%d", len(slot_label_list))

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        print('start to evalutate')
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        predict_features = file_based_convert_examples_to_features(predict_examples, domain_label_list,
                                                                 intent_label_list,slot_label_list, FLAGS.max_seq_length, tokenizer,predict_file, 'test')

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
        result = estimator.predict(input_fn=predict_input_fn)

        output_prediction_file = os.path.join(FLAGS.output_dir, FLAGS.predict_file)
        write_predictions(vocab, predict_examples, predict_features, result, domain_label_list, intent_label_list, slot_label_list, num_actual_predict_examples, output_prediction_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
