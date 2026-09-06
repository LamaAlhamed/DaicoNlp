#!/usr/bin/env bash
set -euo pipefail
# Lab 7: provided-style wrapper. Requires `hey` installed separately.
# TODO: keep request payload aligned with the final /v1/classify schema.
hey -z 60s -c 16 -m POST -H 'Content-Type: application/json' \
  -d '{"text":"الخدمة ممتازة ولكن التأخير طويل"}' \
  http://127.0.0.1:8000/v1/classify
