#!/usr/bin/bash
cd docs
./build.sh
cd ..

rm dist/*
python setup.py sdist bdist_wheel
twine check dist/*
