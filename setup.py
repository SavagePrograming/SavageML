from setuptools import setup

setup(
    name='SavageML',
    version='0.1.1',
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
