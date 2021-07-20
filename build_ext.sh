#!/bin/bash

set -e

SPATH=$(readlink -f $(dirname $0))

cd ${SPATH}
./setup.py build_ext --inplace
