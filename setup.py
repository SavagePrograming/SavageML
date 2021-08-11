from setuptools import setup
from savageml import __version__
setup(
    name='SavageML',
    version=__version__,
    packages=['savageml', 'savageml.models', 'savageml.simulations'],
    url='https://github.com/SavagePrograming/SavageML',
    license='MIT License',
    author='William Savage',
    author_email='savage.programing@gmail.com',
    description='A Personal Experimental Machine Learning Library',
    python_requires='>=3',
    install_requires=[
        "scikit-learn",
        "numpy"
    ]
)
