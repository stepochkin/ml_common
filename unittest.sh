#!/bin/bash

set -e

SPATH=$(readlink -f $(dirname $0))

cd ${SPATH}
python3 $(which nose2) -v $1
