#!/usr/bin/env bash
# Compile ORCA (graphlet orbit counts used by the orbit MMD metric).
set -euo pipefail
cd "$(dirname "$0")/../dig2/metrics/orca"
g++ -O2 -std=c++11 -o orca orca.cpp
echo "built $(pwd)/orca"
