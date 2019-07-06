#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: libo
# @Date  : 2019/6/20
"""Training Wide and Deep Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import shutil
import sys
import time

import tensorflow as tf

from lib.read_conf import Config
from lib.dataset import input_fn
from lib.build_estimator import build_estimator,build_custom_estimator
from lib.util import elapse_time, list_files

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Train Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
parser.add_argument(
    '--train_epochs', type=int, default=CONFIG["train_epochs"],
    help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=CONFIG["epochs_per_eval"],
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default=CONFIG["train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--eval_data', type=str, default=CONFIG["eval_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--test_data', type=str, default=CONFIG["test_data"],
    help='Path to the test data.')
parser.add_argument(
    '--keep_train', type=int, default=CONFIG["keep_train"],
    help='Whether to keep training on previous trained model.')


def train_and_eval(model):
    for n in range(FLAGS.train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        train_data = FLAGS.train_data  # dir to file list
#         for f in train_data_list:
        t0 = time.time()
        tf.logging.info('<EPOCH {}>: Start training'.format(n + 1))
        model.train(
            input_fn=lambda: input_fn(train_data, 'train', FLAGS.batch_size),
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None)
        tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, train_data, elapse_time(t0)))
        print('-' * 80)
        tf.logging.info('<EPOCH {}>: Start evaluating {}'.format(n + 1, FLAGS.eval_data))
        t0 = time.time()
        results = model.evaluate(
            input_fn=lambda: input_fn(FLAGS.eval_data, 'eval', FLAGS.batch_size),
            steps=None,  # Number of steps for which to evaluate model.
            hooks=None,
            checkpoint_path=None,  # latest checkpoint in model_dir is used.
            name=None)
        tf.logging.info('<EPOCH {}>: Finish evaluation {}, take {} mins'.format(n + 1, FLAGS.eval_data, elapse_time(t0)))
        print('-' * 80)
        # Display evaluation metrics
        for key in sorted(results):
            print('{}: {}'.format(key, results[key]))
        # every epochs_per_eval test the model (use larger test dataset)
#         if (n+1) % FLAGS.epochs_per_eval == 0:
#             tf.logging.info('<EPOCH {}>: Start testing {}'.format(n + 1, FLAGS.test_data))
#             results = model.evaluate(
#                 input_fn=lambda: input_fn(FLAGS.eval_data, 'eval', FLAGS.batch_size),
#                  steps=None,  # Number of steps for which to evaluate model.
#                  hooks=None,
#                  checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
#                  name=None)
#             tf.logging.info('<EPOCH {}>: Finish testing {}, take {} mins'.format(n + 1, FLAGS.test_data, elapse_time(t0)))
#             print('-' * 80)
#             # Display evaluation metrics
#             for key in sorted(results):
#                 print('{}: {}'.format(key, results[key]))

def train(model):
    for n in range(FLAGS.train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        train_data_list = list_files(FLAGS.train_data)  # dir to file list
        for f in train_data_list:
            t0 = time.time()
            tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, f))
            model.train(
                input_fn=lambda: input_fn(f, FLAGS.image_train_data, 'train', FLAGS.batch_size),
                hooks=None,
                steps=None,
                max_steps=None,
                saving_listeners=None)
            tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, f, elapse_time(t0)))


def train_and_eval_api(model):
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(FLAGS.train_data,'train', FLAGS.batch_size))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(FLAGS.eval_data,'eval',FLAGS.batch_size))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)            

    
def main(unused_argv):
    CONFIG = Config()
    print("Using TensorFlow Version %s" % tf.__version__)
    assert "1.4" <= tf.__version__, "Need TensorFlow r1.4 or Later."
    print('\nModel Type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('\nModel Directory: {}'.format(model_dir))

    print("\nUsing Train Config:")
    for k, v in CONFIG.train.items():
        print('{}: {}'.format(k, v))
    print("\nUsing Model Config:")
    for k, v in CONFIG.model.items():
        print('{}: {}'.format(k, v))

    if not FLAGS.keep_train:
        # Clean up the model directory if not keep training
        shutil.rmtree(model_dir, ignore_errors=True)
        print('Remove model directory: {}'.format(model_dir))
    model = build_custom_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))
    train_and_eval_api(model)


if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
