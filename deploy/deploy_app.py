import json
import io
import zipfile
import subprocess
import boto3

# aws related
REGION = "eu-central-1"
ENDPOINT_URL = "http://localhost:4566"

AWS_CONFIG = {"region_name": REGION, "endpoint_url": ENDPOINT_URL}

# sagemaker config
SAGEMAKER_CONTAINER_IMAGE_URI = "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.2-cpu-py38-ubuntu20.04-sagemaker"
MODEL_BUCKET_NAME = "mnist-model-bucket"
MNIST_MODEL_LOCATION = "ml/results/zip/model.tar.gz"
MNIST_MODEL_KEY = "model.tar.gz"
MNIST_MODEL_NAME = "mnist-model"
MNIST_EP_CONFIGURATION_NAME = "mnist-epc"
MNIST_EP_NAME = "mnist-endpoint"

# lambda config
LAMBDA_NAME = "MnistHandlerLambda"

# website config
WEBSITE_BUCKET_NAME = "mnist-website"


def create_sagemaker_endpoint():
    # create s3 bucket for model hosting
    s3 = boto3.client("s3", **AWS_CONFIG)
    s3.create_bucket(
        Bucket=MODEL_BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": REGION},
    )

    # upload ml model to s3
    s3.upload_file(
        Filename=MNIST_MODEL_LOCATION,
        Bucket=MODEL_BUCKET_NAME,
        Key=MNIST_MODEL_KEY,
    )

    # create role for sagemaker to access s3
    iam = boto3.client("iam", **AWS_CONFIG)

    document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    sagemaker_role = iam.create_role(
        RoleName="sagemaker-role", AssumeRolePolicyDocument=json.dumps(document)
    )
    iam.attach_role_policy(
        RoleName="sagemaker-role",
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                "Resource": "arn:aws:s3:::*",
            }
        ],
    }
    iam.put_role_policy(
        RoleName="sagemaker-role",
        PolicyName="sagemaker-s3-access",
        PolicyDocument=json.dumps(s3_policy),
    )

    # create sagemaker model and endpoint
    sm = boto3.client("sagemaker", **AWS_CONFIG)

    model_data_url = f"s3://{MODEL_BUCKET_NAME}/model.tar.gz"

    sm.create_model(
        ModelName=MNIST_MODEL_NAME,
        PrimaryContainer={
            "Image": SAGEMAKER_CONTAINER_IMAGE_URI,
            "Mode": "SingleModel",
            "ModelDataUrl": model_data_url,
        },
        ExecutionRoleArn=sagemaker_role["Role"]["Arn"],
    )

    sm.create_endpoint_config(
        EndpointConfigName=MNIST_EP_CONFIGURATION_NAME,
        ProductionVariants=[
            {
                "VariantName": "single-variant",
                "ModelName": MNIST_MODEL_NAME,
                "ServerlessConfig": {
                    "MemorySizeInMB": 6144,
                    "MaxConcurrency": 8,
                },
            },
        ],
    )

    sm.create_endpoint(
        EndpointName=MNIST_EP_NAME,
        EndpointConfigName=MNIST_EP_CONFIGURATION_NAME,
    )


def create_lambda():
    # create function for handling requests which are passed onto sagemaker endpoint
    lambda_client = boto3.client("lambda", **AWS_CONFIG)

    # Zip up the Lambda code from the specified directory
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, mode="w") as zip_file:
            zip_file.write("lambda/index.py", arcname="index.py")
        zip_content = buffer.getvalue()

    # Create the Lambda function
    response = lambda_client.create_function(
        FunctionName=LAMBDA_NAME,
        Runtime="python3.9",
        Role="arn:aws:iam::000000000000:role/lambda-role",
        Handler="index.lambda_handler",
        Code={"ZipFile": zip_content},
        Timeout=300,
        Environment={"Variables": {"SAGEMAKER_ENDPOINT_NAME": MNIST_EP_NAME}},
    )

    response = lambda_client.create_function_url_config(
        FunctionName=LAMBDA_NAME,
        AuthType="NONE",
        Cors={
            "AllowCredentials": True,
            "AllowMethods": [
                "*",
            ],
            "AllowOrigins": [
                "*",
            ],
            "MaxAge": 123,
        },
    )

    return response["FunctionUrl"]


def build_webapp(lambdaUrl: str):
    for env in ["production", "development"]:
        f = open(f"web/.env.{env}", "w")
        f.write(f"REACT_APP_LAMBDA_URL={lambdaUrl}")
        f.close()

    # build web app
    subprocess.run("npm run build", cwd="./web", shell=True)


def host_website():
    # create s3 bucket for hosting the web app
    s3 = boto3.client("s3", **AWS_CONFIG)
    s3.create_bucket(
        Bucket=WEBSITE_BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
    )

    # Set the bucket policy for static website hosting
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{WEBSITE_BUCKET_NAME}/*",
            }
        ],
    }
    s3.put_bucket_policy(Bucket=WEBSITE_BUCKET_NAME, Policy=json.dumps(policy))

    # Upload the website files to the bucket
    subprocess.run(f"awslocal s3 sync web/build s3://{WEBSITE_BUCKET_NAME}", shell=True)

    # Enable static website hosting for the bucket
    s3.put_bucket_website(
        Bucket=WEBSITE_BUCKET_NAME,
        WebsiteConfiguration={
            "ErrorDocument": {"Key": "index.html"},
            "IndexDocument": {"Suffix": "index.html"},
        },
    )

    # Print the URL of the static website
    print(
        f"S3 static website URL: http://{WEBSITE_BUCKET_NAME}.s3-website.localhost.localstack.cloud:4566"
    )


def main():
    create_sagemaker_endpoint()

    lambdaUrl = create_lambda()

    build_webapp(lambdaUrl)

    host_website()


if __name__ == "__main__":
    main()
