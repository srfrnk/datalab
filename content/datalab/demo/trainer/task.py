""" Task """
import argparse
import os

import trainer.experiment

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam


def main():
    """ Main """
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train_files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int,
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )
    parser.add_argument(
        '--test_files',
        help='GCS or local paths to evaluation data',
        nargs='+',
        required=True
    )
    # Training arguments
    parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns',
        default=8,
        type=int
    )
    parser.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=100,
        type=int
    )
    parser.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=4,
        type=int
    )
    parser.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--job_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--md_dir',
        help='Path to metadata',
        required=True
    )

    # Argument to turn on all logging
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    # Experiment arguments
    parser.add_argument(
        '--eval-delay-secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--min-eval-frequency',
        help='Minimum number of training steps between evaluations',
        default=None,  # Use TensorFlow's default (currently, 1000 on GCS)
        type=int
    )
    parser.add_argument(
        '--train_steps',
        help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
        type=int
    )
    parser.add_argument(
        '--eval_steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=100,
        type=int
    )
    parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON'
    )

    args = parser.parse_args()

    tf.logging.set_verbosity(args.verbosity)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    learn_runner.run(
        trainer.experiment.generate_experiment_fn(
            md_dir=args.md_dir,
            train_files=args.train_files,
            test_files=args.test_files,
            min_eval_frequency=args.min_eval_frequency,
            eval_delay_secs=args.eval_delay_secs,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps
        ),
        run_config=run_config.RunConfig(model_dir=args.job_dir),
        hparams=hparam.HParams(**args.__dict__)
    )


if __name__ == '__main__':
    main()
