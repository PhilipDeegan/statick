#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

shell_session_update() { :; }

brew update
brew upgrade pyenv
brew install swig

PYVER=3.6.7

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export CC="clang"
export CXX="clang++"

eval "$(pyenv init -)"
env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s ${PYVER}
pyenv local ${PYVER}
export PATH="/Users/travis/.pyenv/versions/3.6.7/bin:$PATH"
