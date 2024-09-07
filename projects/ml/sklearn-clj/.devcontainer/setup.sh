#!/bin/bash -xe

clj -Sthreads 1 -P
mkdir -p .venv 
poetry install
