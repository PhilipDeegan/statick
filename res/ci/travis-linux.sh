#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/../..
ROOT=$PWD

g++ -march=native -Q --help=target
python3 -V

git clone https://github.com/swig/swig -b rel-4.0.0 swig && \
  cd swig && ./autogen.sh && ./configure --without-pcre && \
  make && sudo make install && cd $ROOT

R="https://github.com/X-DataInitiative/tick --depth 1 tick -b master --recursive"
TM=(array base random base_model linear_model preprocessing robust prox solver)
git clone $R && (cd tick && python3 -m pip install -r requirements.txt --user && \
                 ./sh/mkn.sh ${TM[@]})

KLOG=3 ./test.sh
KLOG=3 mkn build test -tdOa -D_KUL_USE_CBLAS -l "-lblas -pthread" $XTRA

exit 0
