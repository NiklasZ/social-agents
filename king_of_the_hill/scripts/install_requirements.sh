#!/bin/bash
exclude_list=$(cat exclude.txt)
pip install $(grep -v "$exclude_list" requirements.txt)
