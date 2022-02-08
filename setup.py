import sys

from setuptools import setup, find_packages

setup(
    name="tensor2struct",
    version="0.3",
    author='bailin',
    author_email='bailin.wang28@gmail.com',
    description="neural semantic parsers",
    packages=find_packages(
        exclude=["experiments", "experiments.*", "*_test.py", "test_*.py", "tests"]
    ),
    include_package_data=True,
    package_data={"": ["*.asdl", "*.json"]},
    install_requires=[
        "astor~=0.7.1",
        "asdl~=0.1.5",
        "attrs~=18.2.0",
        "babel~=2.7.0",
        "cython~=0.29.1",
        "dacite~=1.2.0",
        "editdistance~=0.5.3",
        "einops~=0.2.0",
        "edit_distance~=1.0.4",
        "higher",
        "jsonnet~=0.11.2",
        "nltk~=3.4",
        "networkx~=2.2",
        "numpy>=1.15",
        "records~=0.5.3",
        "sacrebleu~=1.5.0",
        "sentencepiece~=0.1.95",
        "spacy==2.3.2",
        "scipy~=1.5.2",
        "stanza~=1.0.0",
        "spacy-stanza==0.2.0",
        "tabulate~=0.8.6",
        "timeout-decorator~=0.4.1",
        "tqdm>=4.36.1",
        "transformers>=2.11.0",
        "pytest~=5.4.1",
        "pyrsistent~=0.14.9",
        "unidecode~=1.1.1",
        'vncorenlp~=1.0.3',
    ],
    entry_points={"console_scripts": ["tensor2struct=tensor2struct.commands.run:main"]},
)
