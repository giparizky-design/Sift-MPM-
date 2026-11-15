#!/bin/bash
set -ex
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== MPM Training Instance Bootstrap Started ==="
date

# 1. Install Docker & AWS CLI tools
dnf update -y
dnf install -y docker
systemctl enable --now docker

# Create 4GB swapfile to prevent OOM
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# 2. Configure directories
mkdir -p /opt/ml/input/data/data
mkdir -p /opt/ml/input/data/models
mkdir -p /opt/ml/code
mkdir -p /opt/ml/model
mkdir -p /opt/ml/output/data

# 3. Pull data, models, and code from S3
echo "Syncing assets from S3..."
aws s3 sync s3://sagemaker-ap-southeast-2-355103369883/geospatials/data/ /opt/ml/input/data/data/
aws s3 sync s3://sagemaker-ap-southeast-2-355103369883/geospatials/models/latest/ /opt/ml/input/data/models/
aws s3 cp s3://sagemaker-ap-southeast-2-355103369883/geospatials/code/sourcedir.tar.gz /opt/ml/code/
tar -xzf /opt/ml/code/sourcedir.tar.gz -C /opt/ml/code/

# Copy models into local outputs directory expected by the engine
mkdir -p /opt/ml/code/outputs/seven_pass_east_kalimantan/models
cp -r /opt/ml/input/data/models/* /opt/ml/code/outputs/seven_pass_east_kalimantan/models/ || true

# 4. Log in to AWS Deep Learning Container ECR
echo "Logging in to AWS Deep Learning ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.ap-southeast-2.amazonaws.com

# 5. Run the incremental MPM training container (Round 8)
echo "Starting training container..."
docker run --rm \
  --network host \
  -v /opt/ml:/opt/ml \
  -w /opt/ml/code \
  -e SM_CHANNEL_DATA=/opt/ml/input/data/data \
  -e SM_CHANNEL_MODELS=/opt/ml/input/data/models \
  -e SM_MODEL_DIR=/opt/ml/model \
  -e SM_OUTPUT_DATA_DIR=/opt/ml/output/data \
  763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker-v1.7 \
  bash -c "pip install -r scripts/requirements_sagemaker.txt && python scripts/sagemaker_entry.py --rounds 1 --trees-per-round 50 --bags-per-round 3 --dae-epochs 2"

# 6. Push updated model checkpoints and logs back to S3
echo "Pushing updated weights and history back to S3..."
aws s3 sync /opt/ml/model/ s3://sagemaker-ap-southeast-2-355103369883/geospatials/outputs/round_8/
aws s3 sync /opt/ml/model/ s3://sagemaker-ap-southeast-2-355103369883/geospatials/models/latest/
aws s3 cp /var/log/user-data.log s3://sagemaker-ap-southeast-2-355103369883/geospatials/outputs/round_8/training_instance.log

# 7. (removed in v4.16, audit v4.14 B02/T04) The dashboard server used to be started
#    here with `--host 0.0.0.0 --network host --restart always`, which exposed the
#    train/stop/delete/clean routes to the instance's network. Do not run the dashboard
#    on a public interface. To view it, run it on the instance bound to 127.0.0.1 and
#    tunnel: ssh -L 8765:127.0.0.1:8765 ec2-user@<instance>

echo "=== Setup complete. Training finished; artefacts synced to S3. ==="
date
