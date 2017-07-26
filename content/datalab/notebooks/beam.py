import tempfile
import json

import apache_beam as beam
import tensorflow as tf

import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io


def preprocess():
    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **{
        'job_name': 'test',
        'project': 'srfrnk-test',
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True
    })

    with beam.Pipeline('DirectRunner', options=pipeline_options) as pipeline:
        with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
            text_lines = pipeline | 'read' >> beam.io.ReadFromText(
                './weather.json')

            parsed_lines = text_lines | 'parse' >> beam.Map(
                lambda (text_line): json.loads(text_line))

            raw_data = parsed_lines | 'map' >> beam.Map(lambda (line): {
                'stn': int(line['stn']),
                'year': int(line['year']),
                'mo': int(line['mo']),
                'da': int(line['da']),
                'temp': line['temp'],
            })

            def preprocessing_fn(inputs):
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

            raw_md = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
                'stn': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
                'year': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
                'mo': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
                'da': dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation()),
                'temp': dataset_schema.ColumnSchema(tf.float32, [], dataset_schema.FixedColumnRepresentation())
            }))

            # (trans_data, trans_md), trans_fn = (
            (trans_data, trans_md), _ = (
                (raw_data, raw_md) | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

            trans_data | 'write data' >> beam.io.WriteToTFRecord(
                './data', coder=example_proto_coder.ExampleProtoCoder(trans_md.schema))
            trans_md | 'write metadata' >> beam_metadata_io.WriteMetadata(
                './metadata', pipeline=pipeline)


preprocess()
