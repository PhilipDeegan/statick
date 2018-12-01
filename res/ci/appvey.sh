#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export KLOG=3
curl -o mkn.exe -L https://github.com/Dekken/maiken/raw/binaries/win10_x64/mkn.exe
cp /c/Python36-x64/python.exe /c/Python36-x64/python3.exe
export PY=python3.exe

$PY -V
$PY -m pip install pip --upgrade
$PY -m pip install numpy scipy scikit-learn

export MKN_CL_PREFERRED=1 # forces mkn to use cl even if gcc/clang are found

cd $CWD/../..
./test.sh
