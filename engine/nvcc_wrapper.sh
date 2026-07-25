#!/bin/bash
exec /usr/lib/cuda/bin/nvcc -allow-unsupported-compiler -ccbin /usr/bin/g++-9 "$@"
