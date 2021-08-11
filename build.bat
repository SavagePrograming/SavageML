rd /s /q dist
mkdir dist
python setup.py sdist
python setup.py bdist_wheel --universal
sphinx-build -b html docs/source/ docs/build/html
