#sphinx-build -b html docs/source/ docs/build/html
pushd docs || exit
make clean
make html
popd 