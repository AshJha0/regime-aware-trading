#!/usr/bin/env bash
# Run the end-to-end demo (bash demo.sh from java/, after bash build.sh).
set -euo pipefail
cd "$(dirname "$0")"

java -Dstdout.encoding=UTF-8 -cp out/main com.quant.regime.Demo "$@"
