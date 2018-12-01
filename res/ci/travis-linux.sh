#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PATH="$PWD:/opt/python/3.6.7/bin:$PATH"
KLOG=3 ./test.sh
KLOG=3 mkn build test -tdOa -D_KUL_USE_CBLAS -l "-lblas -pthread" $XTRA
exit 0