#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd $CWD/.. && CWD=$PWD
XTRA=${XTRA:-""}

[ ! -d "$CWD/tick" ] && git clone https://github.com/PhilipDeegan/tick -b releases --depth 2 --recursive --shallow-submodules

(
    cd tick
    mkn -C lib build -dtOg 0 $XTRA
)
