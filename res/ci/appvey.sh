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
$PY -m pip install wheel

R="https://github.com/X-DataInitiative/tick --depth 1 tick -b master --recursive"
TM=(array base random base_model linear_model preprocessing robust prox solver)
git clone $R && cp rc.exe tick/lib && cd tick

git clone https://github.com/X-DataInitiative/tick_appveyor -b master --depth 1 appveyor
cat appveyor/pip/numpy/numpy-1.16.4+mkl-cp37* > appveyor/pip/numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl
$PY -m pip install appveyor/pip/numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl
$PY -m pip install appveyor/pip/numpydoc-0.8.0-py2.py3-none-any.whl
$PY -m pip install appveyor/pip/scipy-1.3.0-cp37-cp37m-win_amd64.whl
$PY -m pip install -r requirements.txt

./sh/mkn.sh ${TM[@]} && cd $ROOT
./test.sh
