#!/usr/bin/bash
./build_docs.sh

rm dist/*
python setup.py sdist bdist_wheel
twine check dist/*
