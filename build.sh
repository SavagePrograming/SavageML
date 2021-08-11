rm -rf ./dist
mkdir ./dist
python setup.py sdist
python setup.py bdist_wheel --universal