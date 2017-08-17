""" Preprocess """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
import tempfile
import pandas as pd
import random

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import textio, tfrecordio
from tensorflow.contrib import learn, lookup
from tensorflow.contrib.layers import feature_column
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io, transform_fn_io
from tensorflow_transform.coders import csv_coder, example_proto_coder
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema, metadata_io


def index_by(key_cols):
    """ index_by """
    stacked = tf.stack(values=key_cols, axis=1)
    concatenated = tf.reduce_join(
        inputs=stacked, reduction_indices=1, separator='_')
    return concatenated


def map_keys_by(key_cols):
    """ map_keys_by """
    key_map = index_by(key_cols=key_cols)
    keys, key_idx, key_counts = tf.unique_with_counts(key_map)
    return key_idx, keys, key_counts, key_map


def group_mean(data, key_cols):
    """ group_mean """
    key_idx, keys, counts, key_map = map_keys_by(key_cols)
    sums = tf.unsorted_segment_sum(
        data=data, segment_ids=key_idx, num_segments=tf.shape(keys)[0])
    means = tf.div(sums, tf.cast(counts, tf.float32))
    return keys, means, key_map


def table_lookup(keys, values, data):
    """ table_lookup """
    table = lookup.HashTable(lookup.KeyValueTensorInitializer(keys, values), -1)
    return table.lookup(data)


def create_preprocessing_fn():
    """ create_preprocessing_fn """

    def preprocessing_fn(inputs):
        """ preprocessing_fn """

        stn = tft.string_to_int(inputs['stn'])
        year = tft.string_to_int(inputs['year'])
        mo = tft.string_to_int(inputs['mo'])
        da = tft.string_to_int(inputs['da'])
        temp = tf.cast(inputs['temp'], tf.float32)

        # mean_mo_key_map = index_by([inputs['project'], inputs['product'], inputs['month']])
        # keys = tf.constant(["{}_{}_{}".format() for row in []])
        # ratios = tf.constant([[] for row in []])
        # mean_mo = tft.apply_function(table_lookup, keys, ratios, key_map)

        # mean_mo_keys, mean_mo_means, mean_mo_key_map = group_mean(temp, [inputs['stn'], inputs['mo']])
        # mean_mo = tft.apply_function(table_lookup, mean_mo_keys, mean_mo_means, mean_mo_key_map)
        
        mean_mo = tft.apply_function(table_lookup,tf.constant(["713570","918310"]) , tf.constant([1.1,2.1]), inputs['stn'])

        # mean_year_keys, mean_year_means, mean_year_key_map = group_mean(temp, [inputs['stn'], inputs['year']])
        # mean_year = tft.apply_function(table_lookup, mean_year_keys, mean_year_means, mean_year_key_map)

        return {
            'stn': stn,
            'year': year,
            'mo': mo,
            'da': da,
            'temp': temp,
            'mean_mo':mean_mo

            # 'mean_mo': mean_mo,
            # 'mean_year': mean_year
        }
    return preprocessing_fn


class ReadData(beam.ptransform.PTransform):
    """ ReadData """

    def __init__(self, files):
        """ __init__ """
        self.files = files

    def expand(self, pcoll):
        """ expand """
        return pcoll | (('read ' + self.files) >> beam.io.ReadFromText(self.files) |
                        ('parse ' + self.files) >> beam.Map(lambda text_line: json.loads(text_line)) |
                        ('map ' + self.files) >> beam.Map(lambda line: {
                            'stn': line['stn'],
                            'year': line['year'],
                            'mo': line['mo'],
                            'da': line['da'],
                            'temp': line['temp']
                        }))


class Sample(beam.ptransform.PTransform):
    """ ReadData """

    def __init__(self, sample_rate):
        """ __init__ """
        self.sample_rate = sample_rate

    def expand(self, pcoll):
        """ expand """
        return pcoll | beam.Filter(lambda item: random.random() < self.sample_rate)


class PartitionTrainTest(beam.ptransform.PTransform):
    """ PartitionTrainTest """

    def __init__(self, num_train, num_test):
        """ __init__ """
        self.num_train = num_train
        self.num_test = num_test

    def expand(self, pcoll):
        """ expand """
        def random_partition_fn(_, num_partitions):
            """ random_partition_fn """
            return int(random.random() * num_partitions)

        num_partitions = self.num_train + self.num_test
        random_partitions = pcoll | ('random partition ' >> beam.Partition(random_partition_fn, num_partitions))
        train_data = [random_partitions[i] for i in range(0, self.num_train)] | 'merge train data' >> beam.Flatten()
        test_data = [random_partitions[i] for i in range(self.num_train, num_partitions)] | 'merge test data' >> beam.Flatten()
        return train_data, test_data


# class Sample(beam.ptransform.PTransform):
#     """ Sample """

#     def __init__(self, num_samples):
#         """ __init__ """
#         self.num_samples = num_samples

#     def expand(self, pcoll):
#         """ expand """
#         return beam.transforms.combiners.Sample.FixedSizeGlobally(pcoll, self.num_samples)


def transform_data(data_files, trans_train_filebase, trans_test_filebase, md_dir):
    """ transform_data """
    raw_md = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
        'stn': dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation()),
        'year': dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation()),
        'mo': dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation()),
        'da': dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation()),
        'temp': dataset_schema.ColumnSchema(tf.float32, [], dataset_schema.FixedColumnRepresentation())
    }))

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **{
        'runner': 'DirectRunner',
        'job_name': 'test',
        'project': 'srfrnk-test',
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    })

    with beam.Pipeline(options=pipeline_options) as pipeline:
        with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
            data = pipeline | ReadData(data_files)
            train_data, test_data = data | PartitionTrainTest(8, 2)

            trans_train_ds, trans_fn = (train_data, raw_md) | beam_impl.AnalyzeAndTransformDataset(create_preprocessing_fn())
            trans_train_data, trans_md = trans_train_ds

            trans_test_ds = ((test_data, raw_md), trans_fn) | beam_impl.TransformDataset()
            trans_test_data, _ = trans_test_ds

            _ = (trans_train_data |
                 'Map to JSON1' >> beam.Map(lambda row: pd.Series(row).to_json()) |
                 'Write Train Text' >> beam.io.WriteToText(file_path_prefix='trans_data/text-train')
                 )
            _ = (trans_test_data |
                 'Map to JSON2' >> beam.Map(lambda row: pd.Series(row).to_json()) |
                 'Write Test Text' >> beam.io.WriteToText(file_path_prefix='trans_data/text-test')
                 )

            _ = trans_train_data | 'Write Train Data' >> tfrecordio.WriteToTFRecord(
                trans_train_filebase, coder=example_proto_coder.ExampleProtoCoder(trans_md.schema))

            _ = trans_test_data | 'Write Test Data' >> tfrecordio.WriteToTFRecord(
                trans_test_filebase, coder=example_proto_coder.ExampleProtoCoder(trans_md.schema))

            _ = (trans_fn | 'Write Transform Function' >> transform_fn_io.WriteTransformFn(md_dir))
            _ = (raw_md | 'Write Raw Metadata' >> beam_metadata_io.WriteMetadata(md_dir, pipeline=pipeline))


def main():
    """ Main """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_dir', help='path to directory containing input data', required=True)

    parser.add_argument('--trans_data_dir', help='path to directory to hold transformed data', required=True)

    parser.add_argument('--data_files', help='pattern for data files', required=True)

    parser.add_argument('--trans_train_data_files', help='pattern for transformed train data files', required=True)
    parser.add_argument('--trans_test_data_files', help='pattern for transformed test data files', required=True)

    parser.add_argument('--md_dir', help='name of metadata dir', required=True)

    args = parser.parse_args()

    data_files = os.path.join(args.input_data_dir, args.data_files)
    trans_train_filebase = os.path.join(args.trans_data_dir, args.trans_train_data_files)
    trans_test_filebase = os.path.join(args.trans_data_dir, args.trans_test_data_files)

    transform_data(data_files, trans_train_filebase, trans_test_filebase, args.md_dir)


if __name__ == '__main__':
    main()
