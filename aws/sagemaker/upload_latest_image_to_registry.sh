# Tag with a timestamp so we have both a copy of the image and a timestamped one
TIMESTAMP=$(date -u +"%Y%m%d%H%M%S")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-west-2

docker tag main:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:latest
docker tag main:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:$TIMESTAMP

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:$TIMESTAMP
