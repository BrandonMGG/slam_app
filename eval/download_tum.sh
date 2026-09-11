#!/usr/bin/env bash
# Descarga y descomprime las secuencias TUM RGB-D usadas en la evaluacion.
# Uso: bash eval/download_tum.sh [dir_destino]   (default: datasets/tum)
set -euo pipefail

DEST="${1:-datasets/tum}"
BASE="https://cvg.cit.tum.de/rgbd/dataset"

SEQS=(
  "freiburg2/rgbd_dataset_freiburg2_pioneer_360"
  "freiburg2/rgbd_dataset_freiburg2_pioneer_slam"
  "freiburg2/rgbd_dataset_freiburg2_desk"
  "freiburg1/rgbd_dataset_freiburg1_xyz"
)

mkdir -p "$DEST"
for seq in "${SEQS[@]}"; do
  name="$(basename "$seq")"
  if [ -d "$DEST/$name" ]; then
    echo "[skip] $name ya existe"
    continue
  fi
  echo "[down] $name"
  wget -q --show-progress -c "$BASE/$seq.tgz" -O "$DEST/$name.tgz"
  tar -xzf "$DEST/$name.tgz" -C "$DEST"
  rm -f "$DEST/$name.tgz"
done
echo "Listo. Secuencias en $DEST/"
