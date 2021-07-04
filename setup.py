from setuptools import setup
from pip.req import parse_requirements

install_reqs = parse_requirements("requirements.txt")
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='SavageML',
    version='0.1.0.3',
    packages=['savageml', 'savageml.models', 'savageml.simulations'],
    url='https://github.com/SavagePrograming/SavageML',
    license='MIT License',
    author='William Savage',
    author_email='savage.programing@gmail.com',
    description='A Personal Experimental Machine Learning Library',
    python_requires='>=3',
    install_requires=[
        "pytest>=6.2.4"
        "scikit-learn>=0.24.2",
        "numpy>=1.21.0"
    ]
)
