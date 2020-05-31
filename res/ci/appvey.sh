#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/../..
ROOT=$PWD

export KLOG=3
curl -o mkn.exe -L https://github.com/Dekken/maiken/raw/binaries/win10_x64/mkn.exe
cp /c/Python37-x64/python.exe /c/Python37-x64/python3.exe
export PY=python3.exe
export MKN_CL_PREFERRED=1 # forces mkn to use cl even if gcc/clang are found


$PY -V
$PY -m pip install pip --upgrade
$PY -m pip install -r res/pip_deps.txt --user --upgrade

./sh/mkn.sh ${TM[@]} && cd $ROOT
export XTRA="-a -std:c++17"
./test.sh
