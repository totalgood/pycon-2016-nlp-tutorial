#!/usr/bin/env sh

# FIXME: find all revealjs markdown files and run pandoc on them like this

BUILD_DIR=`pwd`
rm -rf "$BUILD_DIR/images/"
mkdir -p "$BUILD_DIR/images/"
# cp -f "$BUILD_DIR/../hobson.github.io/images/"* "$BUILD_DIR/images/"
# this seemed to break the github builder
cp -rf "$BUILD_DIR/../hobson.github.io/images" "$BUILD_DIR/"
# cp -f `pwd`/../hobson.github.io/images/*.png `pwd`/images/
# cp -f `pwd`/../hobson.github.io/images/*.svg `pwd`/images/
# cp -f `pwd`/../hobson.github.io/images/*.gif `pwd`/images/

if [ -z "$1" ]; then
    PRESENTATION=2015-10-27-Hacking-Oregon-Hidden-Political-Connections
else
    PRESENTATION="$1"
fi
MARKDOWN="$BUILD_DIR/../hobson.github.io/_posts/${PRESENTATION}.md"
BLOG="$BUILD_DIR/../hobson.github.io/"
HTML="$BUILD_DIR/${PRESENTATION}.html"
IPYNB="$BUILD_DIR/../webapps/hackor/"


pandoc -t revealjs --mathjax --template=`pwd`/pandoc-template-for-revealjs.html -V theme=moon -s "$MARKDOWN" -o "$HTML"
sed -i -e 's/src\=\"\/images/src="\/talks\/images/g' "$HTML"

git add *.html
git add images/
git add notebooks/*.ipynb
git commit -am "auto-commit new build  $(date)"
git push tg
git push origin

git checkout master
git merge gh-pages
git push tg
git push origin

git checkout gh-pages
git merge master
git push tg
git push origin

cd "$BLOG"
git add _posts
git commit -am 'update blog post to reflect slides'
git push
cd "$BUILD_DIR"

cd "$IPYNB"
git add *.ipynb
git add data/*.ipynb
git commit -am 'update ipython notebooks to reflect talk slides'
git push
cd "$BUILD_DIR"
