#!/usr/bin/bash

rm -rf dist
mkdir dist

find ../examples/* -type f -name '*.ipynb' -exec jupyter nbconvert --to html --output-dir dist/examples {} + || exit 1

pdoc --html --template-dir . --force -o dist ../mogptk || exit 1
mv -f dist/mogptk/* dist || exit 1
rmdir dist/mogptk || exit 1

# create examples bootstrap
sed -n '1,/<main>/p' dist/index.html > dist/examples.html
echo '<article id="content" style="padding:0">' >> dist/examples.html
echo '<iframe id="example" width="100%" height="100%"></iframe>' >> dist/examples.html
echo '</article>' >> dist/examples.html
echo '<script>' >> dist/examples.html
echo 'let q = new URLSearchParams(window.location.search).get("q");' >> dist/examples.html
echo 'if (/^[a-zA-Z0-9_-]+$/.test(q)) {' >> dist/examples.html
echo 'document.getElementById("example").src = "examples/" + q + ".html";' >> dist/examples.html
echo '}' >> dist/examples.html
echo '</script>' >> dist/examples.html
sed -n '/<nav id=\"sidebar\">/,$p' dist/index.html >> dist/examples.html
