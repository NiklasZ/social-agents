# Tag with a timestamp so we have both a copy of the image and a timestamped one
TIMESTAMP=$(date -u +"%Y%m%d%H%M%S")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-west-2

docker tag main:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:latest
docker tag main:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:$TIMESTAMP

LATEST_IMAGE=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:latest
TIMESTAMP_IMAGE=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/social-agents:$TIMESTAMP

# This is a lazy "log in" if the token has expired check.
if docker pull $TIMESTAMP_IMAGE 2>&1 | grep -q 'denied: Your authorization token has expired'; then
    echo "Authentication token has expired. Logging in again..."
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 931357261381.dkr.ecr.us-west-2.amazonaws.com
fi

docker push $LATEST_IMAGE
docker push $TIMESTAMP_IMAGE

