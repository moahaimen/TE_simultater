#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
DATASETS_CSV="abilene,geant"
SKIP_EXISTING=1

usage() {
  cat <<USAGE
Usage: $0 [--data_dir PATH] [--datasets abilene,geant] [--force]

Downloads required SNDlib dynamic TM archives and native topology files.
USAGE
}

dataset_archive() {
  case "$1" in
    abilene) echo "directed-abilene-zhang-5min-over-6months-ALL-native.tgz" ;;
    geant) echo "directed-geant-uhlig-15min-over-4months-ALL-native.tgz" ;;
    *) return 1 ;;
  esac
}

dataset_topology() {
  case "$1" in
    abilene) echo "abilene.txt" ;;
    geant) echo "geant.txt" ;;
    *) return 1 ;;
  esac
}

download_with_fallback() {
  local output_path="$1"
  shift

  local urls=("$@")
  for url in "${urls[@]}"; do
    echo "Trying: $url"
    if curl -fL --retry 3 --connect-timeout 10 --max-time 600 "$url" -o "$output_path"; then
      echo "Downloaded: $output_path"
      return 0
    fi
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --datasets)
      DATASETS_CSV="$2"
      shift 2
      ;;
    --force)
      SKIP_EXISTING=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"

ARCHIVE_DIR="$DATA_DIR/raw/archives"
TOPO_DIR="$DATA_DIR/raw/topology"
mkdir -p "$ARCHIVE_DIR" "$TOPO_DIR"

for dataset in "${DATASETS[@]}"; do
  dataset="$(echo "$dataset" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$dataset" ]]; then
    continue
  fi

  archive_name="$(dataset_archive "$dataset")" || {
    echo "Unsupported dataset: $dataset"
    exit 1
  }
  topology_name="$(dataset_topology "$dataset")"

  archive_out="$ARCHIVE_DIR/$archive_name"
  topology_out="$TOPO_DIR/$topology_name"

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$archive_out" ]]; then
    echo "Archive already exists, skipping: $archive_out"
  else
    if ! download_with_fallback "$archive_out" \
      "https://sndlib.put.poznan.pl/download/$archive_name" \
      "https://sndlib.put.poznan.pl/download/sndlib-dynamic/$archive_name" \
      "http://sndlib.zib.de/download/$archive_name" \
      "http://sndlib.zib.de/download/sndlib-dynamic/$archive_name"; then
      echo "Failed to download dynamic archive for dataset '$dataset'."
      echo "Tried mirrors under sndlib.put.poznan.pl and sndlib.zib.de."
      exit 1
    fi
  fi

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$topology_out" ]]; then
    echo "Topology already exists, skipping: $topology_out"
  else
    if ! download_with_fallback "$topology_out" \
      "https://sndlib.put.poznan.pl/download/sndlib-networks-native/$topology_name" \
      "http://sndlib.zib.de/download/sndlib-networks-native/$topology_name"; then
      echo "Failed to download topology file for dataset '$dataset'."
      exit 1
    fi
  fi
done

echo "Download complete. Files are in: $DATA_DIR/raw"
