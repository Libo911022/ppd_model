#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: libo
# @Date  : 2019/6/19
"""Parse data and generate input_fn for tf.estimators"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import abc
import tensorflow as tf

import os
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.build_estimator import _build_model_columns
# from lib.utils import image_preprocessing, vgg_preprocessing


class _CTRDataset(object):
    """Interface for dataset using abstract class"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_file):
        # check file exsits, turn to list so that data_file can be both file or directory.
        assert tf.gfile.Exists(data_file), (
            'data file: {} not found. Please check input data path'.format(data_file))
        if tf.gfile.IsDirectory(data_file):
            data_file_list = [f for f in tf.gfile.ListDirectory(data_file) if not f.startswith('.')]
            data_file = [data_file + '/' + file_name for file_name in data_file_list]
        self._data_file = data_file
        self._conf = Config()
        self._train_conf = self._conf.train
        self._train_epochs = self._train_conf["train_epochs"]
        

    @abc.abstractmethod
    def input_fn(self, mode, batch_size):
        """
        Abstract input function for train or evaluation (with label),
        abstract method must be implemented in subclasses when instantiate.
        Args:
            mode: `train`, `eval` or `pred`
                train for train mode, do shuffle, repeat num_epochs
                eval for eval mode, no shuffle, no repeat
                pred for pred input_fn, no shuffle, no repeat and no label 
            batch_size: Int
        Returns:
            (features, label) 
            `features` is a dictionary in which each value is a batch of values for
            that feature; `labels` is a batch of labels.
        """
        raise NotImplementedError('Calling an abstract method.')


class _CsvDataset(_CTRDataset):
    """A class to parse csv data and build input_fn for tf.estimators"""

    def __init__(self, data_file):
        super(_CsvDataset, self).__init__(data_file)
        self._loss_weight = self._train_conf["loss_weight"]
        self._use_weight = True
        self._multivalue = self._train_conf["multivalue"]
        self._feature = self._conf.get_feature_name()  # all features
        self._feature_used = self._conf.get_feature_name('used')  # used features
        self._feature_unused = self._conf.get_feature_name('unused')  # unused features
        self._feature_conf = self._conf.read_feature_conf()  # feature conf dict
        self._csv_defaults = self._column_to_csv_defaults()
        self._shuffle_buffer_size = self._train_conf["num_examples"]

    def _column_to_csv_defaults(self):
        """parse columns to record_defaults param in tf.decode_csv func
        Return: 
            OrderedDict {'feature name': [''],...}
        """
        csv_defaults = OrderedDict()
        csv_defaults['label'] = [0]  # first label default, empty if the field is must
        for f in self._feature:
            if f in self._feature_conf:  # used features
                conf = self._feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        csv_defaults[f] = [0]
                    else:
                        csv_defaults[f] = ['']
                else:
                    csv_defaults[f] = [0.0]  # 0.0 for float32
            else:  # unused features
                csv_defaults[f] = ['']
        return csv_defaults

    def _parse_csv(self, is_pred=False, field_delim='\t', multivalue_delim=','):
        """Parse function for csv data
        Args:
            is_pred: bool, defaults to False
                True for pred mode, parse input data with label
                False for train or eval mode, parse input data without label
            field_delim: csv fields delimiter, defaults to `\t`
            na_value: use csv defaults to fill na_value
            multivalue: bool, defaults to False
                True for csv data with multivalue features.
                eg:   f1       f2   ...
                    a, b, c    1    ...
                     a, c      2    ...
                     b, c      0    ...
            multivalue_delim: multivalue feature delimiter, defaults to `,`
        Returns:
            feature dict: {feature: Tensor ... }
        """
        if is_pred:
            self._csv_defaults.pop('label')
        csv_defaults = self._csv_defaults
        multivalue = self._multivalue
        loss_weight = tf.constant(list(self._loss_weight.values()))
        use_weight = self._use_weight

        def parser(value):
            """Parse train and eval data with label
            Args:
                value: Tensor("arg0:0", shape=(), dtype=string)
            """
            # `tf.decode_csv` return rank 0 Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
            # na_value fill with record_defaults
            columns = tf.decode_csv(
                value, record_defaults=list(csv_defaults.values()),
                field_delim=field_delim, use_quote_delim=False)
            features = dict(zip(csv_defaults.keys(), columns))
            for f, tensor in list(features.items()):
                if f in self._feature_unused:
                    features.pop(f)  # remove unused features
                    continue
                if multivalue:  # split tensor
                    if isinstance(csv_defaults[f][0], str):
                        # input must be rank 1, return SparseTensor
                        # print(st.values)  # <tf.Tensor 'StringSplit_11:1' shape=(?,) dtype=string>
                        features[f] = tf.string_split([tensor], multivalue_delim).values  # tensor shape (?,)
                    else:
                        features[f] = tf.expand_dims(tensor, 0)  # change shape from () to (1,)
            if is_pred:
                return features
            else:
                labels = features.pop('label')
                if use_weight:
                    pred = labels[0] if multivalue else labels  # pred must be rank 0 scalar
                    features["weight_column"] = tf.gather(loss_weight,[pred])  # padded_batch need rank 1
                return features, labels
        return parser

    def input_fn(self, mode, batch_size):
        assert mode in {'train', 'eval', 'pred'}, (
            'mode must in `train`, `eval`, or `pred`, found {}'.format(mode))
        tf.logging.info('Parsing input csv files: {}'.format(self._data_file))
        # Extract lines from input files using the Dataset API.
        dataset = tf.contrib.data.TextLineDataset(self._data_file)
        dataset = dataset.map(self._parse_csv(is_pred=(mode == 'pred')))
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size, seed=2019)
            dataset = dataset.repeat(self._train_epochs)  # define outside loop

        # dataset = dataset.prefetch(2 * batch_size)
        if self._multivalue:
            padding_dic = {k: [None] for k in self._feature_used}
            if self._use_weight and mode != 'pred':
                padding_dic['weight_column'] = [None]
            padded_shapes = padding_dic if mode == 'pred' else (padding_dic, [None])
            dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        else:
            # batch(): each element tensor must have exactly same shape, change rank 0 to rank 1
            dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()



def input_fn(csv_data_file, mode, batch_size):
    """Combine input_fn for tf.estimators
    Combine both csv and image data; combine both train and pred mode.
    set img_data_file None to use only csv data
    """
    if mode == 'pred':
        features = _CsvDataset(csv_data_file).input_fn(mode, batch_size)
        return features

    else:
        features, label = _CsvDataset(csv_data_file).input_fn(mode, batch_size)
        return features, label


def _input_tensor_test(data_file, batch_size=100):
    """test for categorical_column and cross_column input."""
    sess = tf.InteractiveSession()
    features, labels = _CsvDataset(data_file).input_fn('train', batch_size=batch_size)
    print(features['age'].eval())
    print(features['min_term_6m'].eval())
    tag_list = tf.feature_column.categorical_column_with_hash_bucket('tag_list', 10000)
    min_term_6m = tf.feature_column.categorical_column_with_hash_bucket('min_term_6m', 30)
    cell_provinceXage = tf.feature_column.crossed_column(['cell_province', 'age'], 600)
    for f in [tag_list,min_term_6m,cell_provinceXage]:
        # f_dense = tf.feature_column.indicator_column(f)
        f_embed = tf.feature_column.embedding_column(f, 5)
        input_tensor = tf.feature_column.input_layer(features, [f_embed])
        sess.run(tf.global_variables_initializer())
        # input_tensor = tf.feature_column.input_layer(features, [f_dense])
        print('{} input tensor:\n {}'.format(f, input_tensor.eval()))
    dense_tensor = tf.feature_column.input_layer(features, [min_term_6m, tag_list, cell_provinceXage])
    print('total input tensor:\n {}'.format(sess.run(dense_tensor)))

    wide_columns, deep_columns = _build_model_columns()
    dense_tensor = tf.feature_column.input_layer(features, deep_columns)
    sess.run(tf.global_variables_initializer())  # fix Attempting to use uninitialized value error.
    sess.run(tf.tables_initializer())  # fix Table not initialized error.
    print(sess.run(dense_tensor))

if __name__ == '__main__':
    csv_path = '../../data/train_sample.csv' #_sample.csv
    _input_tensor_test(csv_path)
    sess = tf.InteractiveSession()
    data = input_fn(csv_path, 'train', 1000)
    print(sess.run(data))






