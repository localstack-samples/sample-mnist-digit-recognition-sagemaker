import json
import os
import boto3

ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
sm = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):    
    payload = event['body']
    print("received payload: ", payload, type(payload))

    payload_json = json.loads(payload)
    input_data = payload_json['data']
    
    print("invoking endpoint %s with input: %s of type %s" %(ENDPOINT_NAME, input_data, type(input_data)))
    response = sm.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=input_data,
    )

    response = response['Body'].read().decode('utf-8')
    print("got response: ", response, type(response))

    res = {"result": response}

    return {
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": True,
            "Content-Type": "application/json",
        },
        "statusCode": 200,
        "body": json.dumps(res),
    }
