#!/usr/bin/env bash
set -e
mkdir -p data
curl -fL -C - -o data/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf data/imagenette2-160.tgz -C data
