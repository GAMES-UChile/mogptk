#!/usr/bin/bash
cd docs
find * -type f -name '*.html' -exec rm -fr {} + || exit 1
pdoc --html --template-dir .. --force -o . ../mogptk || exit 1
mv -f mogptk/* . || exit 1
rmdir mogptk || exit 1
