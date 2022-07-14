#!/usr/bin/bash

rm -rf docs
mkdir docs

custom_css='<style>\
body{max-width:100ch}\
pre{overflow-x:scroll}\
<\/style>'

find examples/* -maxdepth 0 -type f -name '*.ipynb' -exec jupyter nbconvert --to html --output-dir docs/examples {} + || exit 1
find docs/examples/* -maxdepth 0 -type f -name '*.html' -exec sed -i "s/<\/head>/$custom_css<\/head>/g" {} + || exit 1

pdoc --html --template-dir . --force -o docs mogptk || exit 1
mv -f docs/mogptk/* docs || exit 1
rmdir docs/mogptk || exit 1

# create examples bootstrap
sed -n '1,/<main>/p' docs/index.html > docs/examples.html
cat >> docs/examples.html <<\EOT
<article id="content" style="padding:0;max-width:100%">
    <iframe id="example" width="100%" height="100%"></iframe>
</article>
<script>
    let q = new URLSearchParams(window.location.search).get("q");
    if (/^[a-zA-Z0-9_-]+$/.test(q)) {
        document.getElementById("example").src = "examples/" + q + ".html";
    }
</script>
EOT
sed -n '/<nav id=\"sidebar\">/,$p' docs/index.html >> docs/examples.html
sed -i 's/href="#/href="index.html#/g' docs/examples.html
