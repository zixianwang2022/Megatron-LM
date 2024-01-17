#!/bin/bash

# mypy --disallow-untyped-defs --disallow-incomplete-defs --ignore-missing-imports --disable-error-code attr-defined megatron/core/models/retro/
mypy \
  --disallow-untyped-defs \
  --disallow-incomplete-defs \
  --ignore-missing-imports \
  --disable-error-code attr-defined \
  --disable-error-code arg-type \
  --disable-error-code call-arg \
  --disable-error-code union-attr \
  --disable-error-code assignment \
  --disable-error-code no-redef \
  --disable-error-code override \
  megatron/core/models/retro/

# eof
