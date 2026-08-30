#!/usr/bin/env bash
set -euo pipefail

readonly DATA_DISK_NAME='contextual-forest-l4-data-20260830'
readonly DATA_DEVICE="/dev/disk/by-id/google-${DATA_DISK_NAME}"
readonly MOUNT_POINT='/mnt/contextual-forest'
readonly HARD_STOP_UNIT='contextual-forest-hard-stop'

mkdir -p "${MOUNT_POINT}"
for _ in $(seq 1 60); do
  if [[ -b "${DATA_DEVICE}" ]]; then
    break
  fi
  sleep 2
done
if [[ ! -b "${DATA_DEVICE}" ]]; then
  echo "Dedicated data disk ${DATA_DEVICE} did not appear" >&2
  exit 1
fi

filesystem_type="$(lsblk -no FSTYPE "${DATA_DEVICE}" | tr -d '[:space:]')"
if [[ -z "${filesystem_type}" ]]; then
  mkfs.ext4 -F -m 0 -L contextual-forest-data "${DATA_DEVICE}"
elif [[ "${filesystem_type}" != 'ext4' ]]; then
  echo "Refusing unexpected filesystem ${filesystem_type} on ${DATA_DEVICE}" >&2
  exit 1
fi

filesystem_uuid="$(blkid -s UUID -o value "${DATA_DEVICE}")"
fstab_entry="UUID=${filesystem_uuid} ${MOUNT_POINT} ext4 defaults,nofail 0 2"
if ! grep -q "UUID=${filesystem_uuid}" /etc/fstab; then
  printf '%s\n' "${fstab_entry}" >> /etc/fstab
fi
mountpoint -q "${MOUNT_POINT}" || mount "${MOUNT_POINT}"
chmod 0777 "${MOUNT_POINT}"

# The local Cloud SDK predates maxRunDuration.  Enforce the same four-hour
# boundary inside the guest on every boot; the project-level $10 billing
# hard-stop remains a second independent guard.
systemctl stop "${HARD_STOP_UNIT}.timer" 2>/dev/null || true
systemctl reset-failed "${HARD_STOP_UNIT}.service" 2>/dev/null || true
systemd-run \
  --unit="${HARD_STOP_UNIT}" \
  --on-active=4h \
  --timer-property=AccuracySec=1min \
  /usr/sbin/shutdown -h now

date --iso-8601=seconds > "${MOUNT_POINT}/last-boot.txt"
nvidia-smi > "${MOUNT_POINT}/nvidia-smi-at-boot.txt" 2>&1 || true
