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
  megatron/core/models/retro/

# eof
