#!/bin/bash
# Check the deployed sample end to end with known digits from tests/fixtures:
# 1. the SageMaker endpoint is InService and predicts each digit when invoked directly
# 2. the Lambda function URL used by the web app returns the same predictions
# 3. the S3 website serves the web app, built with the function URL
set -euo pipefail

REGION=eu-central-1
ENDPOINT_NAME=mnist-endpoint
LAMBDA_NAME=MnistHandlerLambda
WEBSITE_URL=http://mnist-website.s3-website.localhost.localstack.cloud:4566
DIGITS=${DIGITS:-"0 3 5"}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-900}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

awsl() { lstk aws --region "$REGION" "$@"; }

waited=0
while true; do
  status=$(awsl sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --query EndpointStatus --output text)
  [ "$status" = "InService" ] && break
  if [ "$status" = "Failed" ] || [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
    echo "Test failed: endpoint is $status after ${waited}s"
    awsl sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --query FailureReason --output text || true
    exit 1
  fi
  echo "Endpoint is $status, waiting"
  sleep 10
  waited=$((waited + 10))
done
echo "PASS: endpoint $ENDPOINT_NAME is InService"

for digit in $DIGITS; do
  # The web app sends one 28x28 image as a JSON string of shape [1, 28, 28]
  jq -c '[.]' "tests/fixtures/digit-$digit.json" > "$TMP/body.json"
  awsl sagemaker-runtime invoke-endpoint --endpoint-name "$ENDPOINT_NAME" \
    --content-type application/json --body "fileb://$TMP/body.json" "$TMP/out.txt" > /dev/null
  predicted=$(cat "$TMP/out.txt")
  if [ "$predicted" != "$digit" ]; then
    echo "Test failed: endpoint predicted '$predicted' for digit $digit"
    exit 1
  fi
done
echo "PASS: endpoint predicts digits $DIGITS"

FUNCTION_URL=$(awsl lambda get-function-url-config --function-name "$LAMBDA_NAME" --query FunctionUrl --output text)
echo "Function URL: $FUNCTION_URL"
for digit in $DIGITS; do
  jq -c '{data: ([.] | tojson)}' "tests/fixtures/digit-$digit.json" > "$TMP/request.json"
  response=$(curl -s -m 120 -X POST -H 'Content-Type: application/json' --data @"$TMP/request.json" "$FUNCTION_URL")
  if ! echo "$response" | jq -e --arg d "$digit" '.result == $d' > /dev/null 2>&1; then
    echo "Test failed: function URL returned '$response' for digit $digit"
    exit 1
  fi
done
echo "PASS: function URL predicts digits $DIGITS"

page=$(curl -s -m 30 -w '\n%{http_code}' "$WEBSITE_URL/")
status=${page##*$'\n'}
if [ "$status" != "200" ] || [[ "$page" != *'<div id="root">'* ]]; then
  echo "Test failed: website returned HTTP $status"
  exit 1
fi
script=$(echo "$page" | grep -o 'static/js/main\.[a-z0-9]*\.js' | head -1)
if ! curl -s -m 30 "$WEBSITE_URL/$script" | grep -qF "$FUNCTION_URL"; then
  echo "Test failed: $script on the website does not call $FUNCTION_URL"
  exit 1
fi
echo "PASS: website serves the app, built with the function URL"

echo "Test passed: the model predicts digits through SageMaker, Lambda and the website"
