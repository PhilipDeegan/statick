#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

shell_session_update() { :; }
travis_footer() { :; }

export PATH="$PWD:$PATH"
export DYLD_LIBRARY_PATH=/usr/local/lib
KLOG=3 ./test.sh
KLOG=3 mkn build test -tdOa -D_KUL_USE_MKL $XTRA \
    -l "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -pthread -ldl"

exit 0

# brew update
# brew upgrade pyenv
# brew install swig

# PYVER=3.6.7

# export PYENV_ROOT="$HOME/.pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# export CC="clang"
# export CXX="clang++"

# eval "$(pyenv init -)"
# env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s ${PYVER}
# pyenv local ${PYVER}
# export PATH="/Users/travis/.pyenv/versions/3.6.7/bin:$PATH"

# export DYLD_INSERT_LIBRARIES=/Applications/Xcode-10.2.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/10.0.1/lib/darwin/libclang_rt.asan_osx_dynamic.dylib
# mv mkn.yaml mkn.yaml.bk; mv res/ci/travis-osx.mkn.yaml mkn.yaml
# mv test.sh test.sh.bk  ; mv res/ci/travis-osx.test.sh test.sh

# sudo /usr/libexec/locate.updatedb
# locate libmkl
# /usr/local/lib/libmkl_rt.dylib /usr/local/lib/libmkl_intel_lp64.dylib  /usr/local/lib/libmkl_core.dylib