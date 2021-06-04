#!/bin/bash
conda env create --file environment.yml
conda activate diveg
python -m pip install -e .
