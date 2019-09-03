#!/usr/bin/bash
find * ! -name 'gen.sh' -exec rm -fr {} +
pdoc --html --force -o . ../mogptk
mv -f mogptk/* .
rmdir mogptk
