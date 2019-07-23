#!/usr/bin/env bash
set -ex
export MKN_LIB_LINK_LIB=1
PY="${PY:-python3}"; PYGET="get_data.py"; which $PY; $PY -V
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
function finish { cd $CWD; }; trap finish EXIT;
R="https://raw.githubusercontent.com/X-DataInitiative/tick/master/"
SA="tick/dataset/fetch_url_dataset.py"; S1="${R}${SA}";
SB="tick/dataset/download_helper.py";   S2="${R}${SB}";
ARF="adult.features.cereal"; ARL="adult.labels.cereal"
URF="url.features.cereal";   URL="url.labels.cereal"
mkn clean build -dStOp statick.py $XTRA; ls -l .
mkdir -p tick/dataset; touch tick/__init__.py; touch tick/dataset/__init__.py;
[ ! -f "${SA}" ] && curl "${S1}" -Lo ${SA}; [ ! -f "${SB}" ] && curl "${S2}" -Lo ${SB}
cat > $CWD/${PYGET} << EOL
import statick
from tick.dataset.download_helper import fetch_tick_dataset
adult_train_set = fetch_tick_dataset('binary/adult/adult.trn.bz2')
from tick.dataset.fetch_url_dataset import fetch_url_dataset
url_10_days = fetch_url_dataset(10, '$CWD')
statick.save_double_sparse2d(adult_train_set[0], "${ARF}")
statick.save_double_array(adult_train_set[1], "${ARL}")
statick.save_double_sparse2d(url_10_days[0], "${URF}")
statick.save_double_array(url_10_days[1], "${URL}")
EOL
PYTHONPATH=$CWD $PY $CWD/${PYGET}
cd $CWD;
mkn clean build test -O $XTRA
