""" Preprocess """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
import tempfile

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


def preprocessing_fn(inputs):
    """ preprocessing_fn """
    stn = inputs['stn']
    year = inputs['year']
    mo = inputs['mo']
    da = inputs['da']
    temp = inputs['temp']

    # stn = stn - tft.mean(stn)
    #   y_normalized = tft.scale_to_0_1(y)
    #   s_integerized = tft.string_to_int(s)
    #   x_centered_times_y_normalized = (x_centered * y_normalized)
    return {
        'stn': stn,
        'year': year,
        'mo': mo,
        'da': da,
        'temp': temp
    }


def get_data(pipeline, files):
    """ get_data """
    return (pipeline |
            ('read '+files) >> beam.io.ReadFromText(files) |
            ('parse '+files) >> beam.Map(lambda text_line: json.loads(text_line)) |
            ('map '+files) >> beam.Map(lambda line: {
                'stn': int(line['stn']),
                'year': int(line['year']),
                'mo': int(line['mo']),
                'da': int(line['da']),
                'temp': line['temp'],
            }))


def transform_data(train_data_files, test_data_files, trans_train_filebase, trans_test_filebase, md_dir):
    """ transform_data """
    raw_md = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
        'stn': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
        'year': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
        'mo': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
        'da': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
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
            train_data = get_data(pipeline, train_data_files)
            trans_train_ds, trans_fn = (train_data, raw_md) | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn)
            trans_train_data, trans_md = trans_train_ds

            test_data = get_data(pipeline, test_data_files)
            trans_test_ds = ((test_data,raw_md), trans_fn) | beam_impl.TransformDataset()
            trans_train_data, _ = trans_test_ds

            _ = trans_train_data | 'Write Train Data' >> tfrecordio.WriteToTFRecord(
                trans_train_filebase, coder=example_proto_coder.ExampleProtoCoder(trans_md.schema))

            _ = trans_train_data | 'Write Test Data' >> tfrecordio.WriteToTFRecord(
                trans_test_filebase, coder=example_proto_coder.ExampleProtoCoder(trans_md.schema))

            _ = (trans_fn | 'Write Transform Function' >> transform_fn_io.WriteTransformFn(md_dir))
            _ = (raw_md | 'Write Raw Metadata' >> beam_metadata_io.WriteMetadata(md_dir, pipeline=pipeline))


def main():
    """ Main """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_dir', help='path to directory containing input data', required=True)

    parser.add_argument('--trans_data_dir', help='path to directory to hold transformed data', required=True)

    parser.add_argument('--train_data_files', help='pattern for train data files', required=True)
    parser.add_argument('--test_data_files', help='pattern for test data files', required=True)

    parser.add_argument('--trans_train_data_files', help='pattern for transformed train data files', required=True)
    parser.add_argument('--trans_test_data_files', help='pattern for transformed test data files', required=True)

    parser.add_argument('--md_dir', help='name of metadata dir', required=True)

    args = parser.parse_args()

    train_data_file = os.path.join(args.input_data_dir, args.train_data_files)
    test_data_file = os.path.join(args.input_data_dir, args.test_data_files)
    trans_train_filebase = os.path.join(args.trans_data_dir, args.trans_train_data_files)
    trans_test_filebase = os.path.join(args.trans_data_dir, args.trans_test_data_files)

    transform_data(train_data_file, test_data_file, trans_train_filebase, trans_test_filebase, args.md_dir)


if __name__ == '__main__':
    main()
