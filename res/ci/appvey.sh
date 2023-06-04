#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/../..
ROOT=$PWD

curl -o mkn.exe -L https://github.com/Dekken/maiken/raw/binaries/win10_x64/mkn.exe

export PATH="$PWD:$PATH"
export KLOG=3
export PY=python3.exe
export MKN_CL_PREFERRED=1 # forces mkn to use cl even if gcc/clang are found

$PY -V
$PY -m pip install pip --upgrade
$PY -m pip install -r res/pip_deps.txt --user --upgrade

export XTRA="-a -std:c++17"
./test.sh
