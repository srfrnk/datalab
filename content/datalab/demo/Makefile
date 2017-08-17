BUCKET=gs://<BUCKET_NAME>
TIMESTAMP=$(shell date +'%s')

setup:
	# gcloud config set project <PROJECTID>
	pip install --user -r requirements.txt

tensorboard-local:
	tensorboard --logdir model

tensorboard-cloud:
	tensorboard --logdir ${BUCKET}/model

preprocess:
	-rm -rf metadata
	-rm -rf trans_data
	python preprocessor/preprocess.py \
		--input_data_dir data \
		--trans_data_dir trans_data \
		--md_dir metadata \
		--data_files data.json \
		--trans_train_data_files train \
		--trans_test_data_files test

train-local:
	-rm -rf model
	gcloud ml-engine local train \
		--module-name trainer.task \
		--package-path trainer/ \
		-- \
		--train_files 'trans_data/train-*' \
		--test_files 'trans_data/test-*' \
		--md_dir metadata \
		--job_dir model \
		--train_steps 1000 \
		--eval_steps 1 \
		--verbosity DEBUG

train-cloud-single:
	-gsutil rm -r ${BUCKET}/model
	gsutil cp -r data ${BUCKET}/data
	gcloud ml-engine jobs submit training train_single_${TIMESTAMP} \
		--job-dir ${BUCKET}/model \
		--runtime-version 1.2 \
		--module-name trainer.task \
		--package-path trainer/ \
		--region us-east1 \
		-- \
		--train-files ${BUCKET}/data/train.data.csv \
		--eval-files ${BUCKET}/data/test.data.csv \
		--train-steps 1000 \
		--eval-steps 100 \
		--verbosity DEBUG
	-gcloud ml-engine jobs stream-logs train_single_${TIMESTAMP}

train-cloud-dist:
	-gsutil rm -r ${BUCKET}/model
	gsutil cp -r data ${BUCKET}/data
	gcloud ml-engine jobs submit training train_dist_${TIMESTAMP} \
		--job-dir ${BUCKET}/model \
		--runtime-version 1.2 \
		--module-name trainer.task \
		--package-path trainer/ \
		--region us-east1 \
		--scale-tier STANDARD_1 \
		-- \
		--train-files ${BUCKET}/data/train.data.csv \
		--eval-files ${BUCKET}/data/test.data.csv \
		--train-steps 1000 \
		--eval-steps 100 \
		--verbosity DEBUG
	-gcloud ml-engine jobs stream-logs train_dist_${TIMESTAMP}

train-cloud-dist-hypertune:
	-gsutil rm -r ${BUCKET}/model
	gsutil cp -r data ${BUCKET}/data
	gcloud ml-engine jobs submit training train_dist_hypertune_${TIMESTAMP} \
		--job-dir ${BUCKET}/model \
		--runtime-version 1.2 \
		--module-name trainer.task \
		--package-path trainer/ \
		--region us-east1 \
		--scale-tier STANDARD_1 \
		--config hptuning_config.yaml \
		-- \
		--train-files ${BUCKET}/data/train.data.csv \
		--eval-files ${BUCKET}/data/test.data.csv \
		--train-steps 1000 \
		--eval-steps 1 \
		--verbosity DEBUG
	-gcloud ml-engine jobs stream-logs train_dist_hypertune_${TIMESTAMP}

deploy-cloud:
	-gcloud ml-engine models create reoptimize --regions=us-east1
	gcloud ml-engine versions create v${TIMESTAMP} --model reoptimize --runtime-version 1.2 --origin $(shell gsutil ls ${BUCKET}/model/export/Servo/ | tail --lines=1)
		
predict-local:
	gcloud ml-engine local predict --format=json --model-dir=model/export/Servo/$(shell ls model/export/Servo/ | tail --lines=1) --json-instances=data/predict.json

predict-cloud:
	gcloud ml-engine predict --format=json --model reoptimize --json-instances data/test.json 
