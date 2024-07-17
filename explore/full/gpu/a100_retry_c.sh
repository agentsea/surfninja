#!/bin/bash

# Maximum number of attempts
MAX_ATTEMPTS=10000
# Time to wait between attempts in seconds
WAIT_TIME=30
# Counter for attempts
ATTEMPTS=0

# Function to execute the gcloud command
create_instance() {
    gcloud compute instances create intern-full-off \
    --project=agentsea-dev \
    --zone=us-central1-c \
    --machine-type=a2-ultragpu-4g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=514070775510-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=4,type=nvidia-a100-80gb \
    --create-disk=auto-delete=yes,boot=yes,device-name=intern-full-off,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240701,mode=rw,size=500,type=projects/agentsea-dev/zones/us-central1-a/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
}

# Loop until the command succeeds or the maximum number of attempts is reached
until create_instance; do
    ATTEMPTS=$((ATTEMPTS+1))
    if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
        echo "Max attempts reached. Exiting."
        exit 1
    fi
    echo "Attempt $ATTEMPTS failed. Retrying in $WAIT_TIME seconds..."
    sleep $WAIT_TIME
done

echo "Instance created successfully."