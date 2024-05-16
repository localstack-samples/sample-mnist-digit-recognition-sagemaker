from aws_cdk import (
    Duration,
    Stack,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    aws_s3_deployment as s3deploy,
    aws_iam as iam,
    RemovalPolicy,
)

from .constants import SAGEMAKER_ENDPOINT_NAME, INFERENCE_IMAGE
from constructs import Construct

class CdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # create bucket which will store machine learning model tar gz
        bucket = s3.Bucket(
            self, "CdkBucket",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # upload file to bucket
        s3deploy.BucketDeployment(
            self, 
            'MlModel', 
            sources=[s3deploy.Source.asset('../ml/results/zip')], 
            destination_bucket=bucket
        )

        sagemaker_role = iam.Role(self, 'sagemaker-role', 
                assumed_by=iam.ServicePrincipal('sagemaker.amazonaws.com'))
        sagemaker_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSageMakerFullAccess'))
        sagemaker_role.add_to_policy(iam.PolicyStatement(
            resources=['arn:aws:s3:::*'],
            actions=['s3:GetObject', 
                    's3:PutObject', 
                    's3:DeleteObject', 
                    's3:ListBucket']))

        # create model
        model = sagemaker.CfnModel(
            self,
            "MLInferenceModel",
            model_name="my-model",
            execution_role_arn=sagemaker_role.role_arn,
            containers=[{
                "image": INFERENCE_IMAGE,
                "mode": "SingleModel",
                "modelDataUrl": f"s3://{bucket.bucket_name}/model.tar.gz"
            }]
        )

        # create endpoint configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "EndpointConfig",
            endpoint_config_name="my-endpoint-config",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    initial_variant_weight=1,
                    model_name=model.model_name,
                    variant_name="single-variant",
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                        max_concurrency=8,
                        memory_size_in_mb=3072,
                    ),
                ),
            ],
        )

        # create endpoint
        endpoint = sagemaker.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_name=SAGEMAKER_ENDPOINT_NAME,
            endpoint_config_name=endpoint_config.endpoint_config_name,
        )    


        # create lambda from code that is in the lambda folder and add it to the stack
        lambda_code = lambda_.Code.from_asset("../lambda")
        ep_lambda = lambda_.Function(
            self, "MnistHandlerLambda",
            code=lambda_code,
            handler="index.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_7,
            timeout=Duration.seconds(300),
            environment={
                SAGEMAKER_ENDPOINT_NAME: endpoint.endpoint_name,
            }
        )

        # # add permission to lambda to invoke sagemaker endpoint
        ep_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint"],
                resources=[endpoint.attr_endpoint_name],
            )
        )


