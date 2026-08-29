#!/usr/bin/env bash
# Run the end-to-end demo (build.sh first). Must be run from java/.
set -euo pipefail
cd "$(dirname "$0")"

java -Dstdout.encoding=UTF-8 -cp out/main com.quant.regime.Demo "$@"
