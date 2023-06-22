# MNIST handwritten digit recognition model running on a local SageMaker endpoint

| Key          | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Environment  | <img src="https://img.shields.io/badge/LocalStack-deploys-4D29B4.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAKgAAACoABZrFArwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAALbSURBVHic7ZpNaxNRFIafczNTGIq0G2M7pXWRlRv3Lusf8AMFEQT3guDWhX9BcC/uFAr1B4igLgSF4EYDtsuQ3M5GYrTaj3Tmui2SpMnM3PlK3m1uzjnPw8xw50MoaNrttl+r1e4CNRv1jTG/+v3+c8dG8TSilHoAPLZVX0RYWlraUbYaJI2IuLZ7KKUWCisgq8wF5D1A3rF+EQyCYPHo6Ghh3BrP8wb1en3f9izDYlVAp9O5EkXRB8dxxl7QBoNBpLW+7fv+a5vzDIvVU0BELhpjJrmaK2NMw+YsIxunUaTZbLrdbveZ1vpmGvWyTOJToNlsuqurq1vAdWPMeSDzwzhJEh0Bp+FTmifzxBZQBXiIKaAq8BBDQJXgYUoBVYOHKQRUER4mFFBVeJhAQJXh4QwBVYeHMQJmAR5GCJgVeBgiYJbg4T8BswYPp+4GW63WwvLy8hZwLcd5TudvBj3+OFBIeA4PD596nvc1iiIrD21qtdr+ysrKR8cY42itCwUP0Gg0+sC27T5qb2/vMunB/0ipTmZxfN//orW+BCwmrGV6vd63BP9P2j9WxGbxbrd7B3g14fLfwFsROUlzBmNM33XdR6Meuxfp5eg54IYxJvXCx8fHL4F3w36blTdDI4/0WREwMnMBeQ+Qd+YC8h4g78wF5D1A3rEqwBiT6q4ubpRSI+ewuhP0PO/NwcHBExHJZZ8PICI/e73ep7z6zzNPwWP1djhuOp3OfRG5kLROFEXv19fXP49bU6TbYQDa7XZDRF6kUUtEtoFb49YUbh/gOM7YbwqnyG4URQ/PWlQ4ASllNwzDzY2NDX3WwioKmBgeqidgKnioloCp4aE6AmLBQzUExIaH8gtIBA/lFrCTFB7KK2AnDMOrSeGhnAJSg4fyCUgVHsolIHV4KI8AK/BQDgHW4KH4AqzCQwEfiIRheKKUAvjuuu7m2tpakPdMmcYYI1rre0EQ1LPo9w82qyNziMdZ3AAAAABJRU5ErkJggg=="> |
| Services     | S3, SageMaker, Lambda                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Integrations | AWS SDK                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Categories   | Serverless, S3 website, Lambda function URLs, SageMaker, Machine Learning, JavaScript, Python                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Level        | Intermediate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

## Introduction

This is a sample application that demonstrates how to use SageMaker on LocalStack.
A simple web frontend allows users to draw a digit and submit it to a locally running SageMaker endpoint.
The endpoint returns a prediction of the digit, which is then displayed in the web frontend.
Request handling is performed by a Lambda function, accessible via a function URL, that uses the SageMaker SDK to invoke the endpoint.

Here's a short summary of AWS service features we use:
* S3 website
* Lambda function URLs
* SageMaker endpoint

Here's the web application in action:

https://user-images.githubusercontent.com/39307517/234888629-4bd9deb8-ecdd-46a6-91d6-908b9f2a443c.mov

## Architecture overview

![Architecture Diagram](/assets/architecture-diagram.png?raw=True "Architecture Diagram")


## Prerequisites

### Dev environment

Create a virtualenv and install all the development dependencies there:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you'd like to perform training locally, you'll need to install the ml dev dependencies as well:

```bash
pip install -r ml/requirements.txt
```

You'll also need npm/node installed to build the web application. Please install according to official guidelines: https://github.com/nvm-sh/nvm

### Download pytorch container image
As our inference container, we use the PyTorch inference container from the AWS ECR.

```bash
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.eu-central-1.amazonaws.com
docker pull 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.2-cpu-py38-ubuntu20.04-sagemaker
```

### LocalStack

Start LocalStack Pro with your API key:

```bash
LOCALSTACK_API_KEY=... localstack start
```

## Instructions

First, we install the dependencies for the Web application in the `web` directory:
```
(cd web; npm install)
```

You can then create the AWS infrastructure on LocalStack by running the `deploy/deploy_app.py` script (make sure to have the virtual environment activated):
```
source .venv/bin/activate
python deploy/deploy_app.py
```

This script will create the SageMaker endpoint with the model, which it first uploads to a bucket.
The script will also create a lambda function that will be used to invoke the endpoint.
Finally, the script will build the web application and then create a s3 website to host it.

### Using the application

Once deployed, visit http://mnist-website.s3-website.localhost.localstack.cloud:4566

Draw something in the canvas and click on the button that says `Predict`.

After a few moments the resulting prediction should be displayed in the box to the right.

![Demo Picture](/assets/demo-pic.png?raw=True "Demo Picture")


## Serverless SageMaker Endpoint

To switch to a serverless SageMaker endpoint you can also execute the deployment script with the additional `-s` or `--serverless` flag:

```bash
python deploy/deploy_app.py --serverless
```

## License

The code of this sample application is published under the Apache 2.0 license (see `LICENSE`).

## Contributing

We appreciate your interest in contributing to our project and are always looking for new ways to improve the developer experience. We welcome feedback, bug reports, and even feature ideas from the community.
Please refer to the [contributing file](CONTRIBUTING.md) for more details on how to get started. 

