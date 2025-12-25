#! /usr/bin/bash
set -e

export PYTHONPATH=$(dirname $0)/..
mypy --package xtuner.v1
