mkdir "./Subprojects/$1"
touch "./Subprojects/$1/__init__.py"
mkdir "./Subprojects/$1/models"
touch "./Subprojects/$1/models/__init__.py"
git add "./Subprojects/$1/models/__init__.py"
mkdir "./Subprojects/$1/simulations"
touch "./Subprojects/$1/simulations/__init__.py"
git add "./Subprojects/$1/simulations/__init__.py"
mkdir "./Subprojects/$1/Tests"
touch "./Subprojects/$1/Tests/__init__.py"
git add "./Subprojects/$1/Tests/__init__.py"
echo "$1 README" > "./Subprojects/$1/README.md"
git add "./Subprojects/$1/README.md"
git commit -m "Initializing $1"