export AWS_ACCESS_KEY_ID ?= test
export AWS_SECRET_ACCESS_KEY ?= test
export AWS_DEFAULT_REGION=us-east-1
SHELL := /bin/bash

usage:		## Show this help
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$//' | sed -e 's/##//'

# The PyTorch inference image the model uses, and its public ECR mirror
INFERENCE_IMAGE ?= 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker
INFERENCE_IMAGE_PUBLIC ?= public.ecr.aws/deep-learning-containers/pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker

install:	## Install dependencies
	@which lstk || npm install -g @localstack/lstk
	@pip install -r requirements.txt
	@cd web && npm ci

pull-image:	## Pull the PyTorch inference image from ECR Public, tagged as the image the model uses
	docker pull $(INFERENCE_IMAGE_PUBLIC)
	docker tag $(INFERENCE_IMAGE_PUBLIC) $(INFERENCE_IMAGE)

start:		## Start LocalStack
	@test -n "${LOCALSTACK_AUTH_TOKEN}" || (echo "LOCALSTACK_AUTH_TOKEN is not set. Find your token at https://app.localstack.cloud/workspace/auth-token"; exit 1)
	@LOCALSTACK_PERSISTENCE=1 LOCALSTACK_AUTH_TOKEN=$(LOCALSTACK_AUTH_TOKEN) lstk start

stop:		## Stop LocalStack
	@lstk stop

deploy:		## Deploy the model, the Lambda function and the website
	python deploy/deploy_app.py $(DEPLOY_ARGS)

test:		## Predict known digits through the endpoint, the function URL and the website
	./test.sh

logs:		## Save the logs in a separate file
	@lstk logs > logs.txt

.PHONY: usage install pull-image start stop deploy test logs
