#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT='interactive-training-2026'
readonly REGION='us-east4'
readonly ZONE='us-east4-c'
readonly NETWORK='contextual-forest-net-20260830'
readonly SUBNET='contextual-forest-subnet-20260830'
readonly FIREWALL='contextual-forest-iap-ssh-20260830'
readonly ROUTER='contextual-forest-router-20260830'
readonly NAT='contextual-forest-nat-20260830'
readonly DATA_DISK='contextual-forest-l4-data-20260830'
readonly VM='contextual-forest-l4-20260830'
readonly IMAGE='ubuntu-accelerator-2204-amd64-with-nvidia-580-v20260825'
readonly IMAGE_PROJECT='ubuntu-os-accelerator-images'
readonly STARTUP_SCRIPT="$(cd "$(dirname "$0")" && pwd)/contextual_forest_startup.sh"

require_absent() {
  local description="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "Refusing to reuse existing resource: ${description}" >&2
    exit 2
  fi
}

require_absent "network ${NETWORK}" \
  gcloud compute networks describe "${NETWORK}" --project="${PROJECT}"
require_absent "subnet ${SUBNET}" \
  gcloud compute networks subnets describe "${SUBNET}" \
  --project="${PROJECT}" --region="${REGION}"
require_absent "firewall ${FIREWALL}" \
  gcloud compute firewall-rules describe "${FIREWALL}" --project="${PROJECT}"
require_absent "router ${ROUTER}" \
  gcloud compute routers describe "${ROUTER}" \
  --project="${PROJECT}" --region="${REGION}"
require_absent "disk ${DATA_DISK}" \
  gcloud compute disks describe "${DATA_DISK}" \
  --project="${PROJECT}" --zone="${ZONE}"
require_absent "instance ${VM}" \
  gcloud compute instances describe "${VM}" \
  --project="${PROJECT}" --zone="${ZONE}"

gcloud compute networks create "${NETWORK}" \
  --project="${PROJECT}" \
  --subnet-mode=custom \
  --bgp-routing-mode=regional \
  --mtu=1460
gcloud compute networks subnets create "${SUBNET}" \
  --project="${PROJECT}" \
  --network="${NETWORK}" \
  --region="${REGION}" \
  --range=10.206.0.0/24 \
  --enable-private-ip-google-access
gcloud compute firewall-rules create "${FIREWALL}" \
  --project="${PROJECT}" \
  --network="${NETWORK}" \
  --direction=INGRESS \
  --priority=1000 \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=contextual-forest-iap
gcloud compute routers create "${ROUTER}" \
  --project="${PROJECT}" \
  --network="${NETWORK}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT}" \
  --project="${PROJECT}" \
  --router="${ROUTER}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --enable-logging \
  --log-filter=ERRORS_ONLY

gcloud compute disks create "${DATA_DISK}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --type=pd-balanced \
  --size=200GB \
  --labels=experiment=contextual-forest,gate=g1,created=20260830

gcloud compute instances create "${VM}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --no-restart-on-failure \
  --image="${IMAGE}" \
  --image-project="${IMAGE_PROJECT}" \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --disk=name="${DATA_DISK}",device-name="${DATA_DISK}",mode=rw,boot=no,auto-delete=no \
  --subnet="${SUBNET}" \
  --no-address \
  --tags=contextual-forest-iap \
  --labels=experiment=contextual-forest,gate=g1,created=20260830 \
  --metadata=enable-oslogin=TRUE \
  --metadata-from-file=startup-script="${STARTUP_SCRIPT}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

gcloud compute instances describe "${VM}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --format='yaml(name,status,machineType,scheduling,disks,networkInterfaces)'
