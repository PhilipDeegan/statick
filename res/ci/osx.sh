#!/usr/bin/env bash
set -e

# shell_session_update() { :; }

XTRA=${XTRA:-""}

export PATH="$PWD:$PATH"
export DYLD_LIBRARY_PATH=/usr/local/lib:/System/Library/Frameworks/ImageIO.framework/Versions/A/Resources

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CWD/../..
ROOT=$PWD

cd /System/Library/Frameworks/ImageIO.framework/Versions/A/Resources
sudo ln -sf libJPEG.dylib /usr/local/lib/libJPEG.dylib
sudo ln -sf libPng.dylib /usr/local/lib/libPng.dylib
sudo ln -sf libTIFF.dylib /usr/local/lib/libTIFF.dylib
sudo ln -sf libGIF.dylib /usr/local/lib/libGIF.dylib
cd $ROOT

export XTRA
KLOG=3 ./sh/build_tick.sh
KLOG=3 ./sh/test.sh
KLOG=3 mkn build test -tdOa -D_KUL_USE_MKL $XTRA \
    -l "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -pthread -ldl"

exit 0
