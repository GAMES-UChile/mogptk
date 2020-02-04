#!/usr/bin/bash
rm dist/*
python setup.py sdist bdist_wheel
twine check dist/*
