#!/usr/bin/bash
./build_docs.sh

rm -f dist/*
python setup.py sdist bdist_wheel
twine check dist/*
