# Reference Vault

## Setup environment

As usual setup your virtual environment:

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip setuptools
$ pip install -r requirements-dev.txt
$ pip install -e .
```

## Basic code compliance

```
$ black .
All done! ✨ 🍰 ✨
X files left unchanged.
$ mypy --ignore-missing-imports .
Success: no issues found in X source files
```

## Run tests

```
$ pytest
```

## Clean up environment

To make sure you have a pristine environment, you might re-create it from time to time. To remove every trace of the virtual environment, run:

```
$ deactivate # if you are still in the virtual environment
$ git clean -fdx -n # dry run
$ git clean -fdx # clean up
```
