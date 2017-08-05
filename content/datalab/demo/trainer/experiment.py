""" Experiment """

import os.path as path
import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import textio
from apache_beam.io import tfrecordio
from tensorflow.contrib import learn
from tensorflow.contrib import lookup
from tensorflow.contrib.layers import feature_column
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import csv_coder
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


def generate_experiment_fn(md_dir, train_files, test_files, **experiment_args):
    """ generate_experiment_fn  """

    def load_md(run_config):
        raw_md = metadata_io.read_metadata(md_dir)
        trans_md = metadata_io.read_metadata(path.join(md_dir, 'transformed_metadata'))
        return raw_md, trans_md

    def experiment_fn(run_config, hparams):
        """ _experiment_fn """
        # num_epochs can control duration if train_steps isn't
        # passed to Experiment

        raw_md, trans_md = load_md(run_config)

        feature_columns = [
            feature_column.one_hot_column(feature_column.sparse_column_with_integerized_feature('stn',bucket_size=1000000)),
            feature_column.real_valued_column('year'),
            feature_column.real_valued_column('mo'),
            feature_column.real_valued_column('da')
        ]

        # estimator = learn.LinearRegressor(
        # feature_columns=feature_columns,
        # model_dir=run_config.model_dir
        # )

        estimator = learn.DNNRegressor(
            hidden_units=[100,100,100],
            feature_columns=feature_columns,
            model_dir=run_config.model_dir            
        )

        train_input_fn = input_fn_maker.build_training_input_fn(
            metadata=trans_md,
            file_pattern=train_files,
            training_batch_size=1000,
            label_keys=['temp'])

        eval_input_fn = input_fn_maker.build_training_input_fn(
            metadata=trans_md,
            file_pattern=test_files,
            training_batch_size=1,
            label_keys=['temp'])

        serve_input_fn = input_fn_maker.build_default_transforming_serving_input_fn(
            raw_metadata=raw_md,
            transform_savedmodel_dir=path.join(md_dir, 'transform_fn'),
            raw_label_keys=['temp'],
            raw_feature_keys=['stn', 'year', 'mo', 'da']
        )

        return tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            export_strategies=[saved_model_export_utils.make_export_strategy(
                serve_input_fn,
                exports_to_keep=1,
                default_output_alternative_key=None,
            )],
            **experiment_args
        )
    return experiment_fn
