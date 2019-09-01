#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
[ -z "$PYTHONPATH" ] && export PYTHONPATH="$CWD/tick:$CWD"
export MKN_LIB_LINK_LIB=1
PY="${PY:-python3}"; PYGET="get_data.py"; which $PY; $PY -V
function finish { cd $CWD; }; trap finish EXIT;
R="https://github.com/X-DataInitiative/tick --depth 1 tick -b master --recursive"
ARF="adult.features.cereal"; ARL="adult.labels.cereal"; URF="url.features.cereal"; URL="url.labels.cereal";
TM=(array base random base_model linear_model preprocessing robust prox solver)
[ ! -d "$CWD/tick" ] && git clone $R && cd tick && ./sh/mkn.sh ${TM[@]} && cd $CWD
mkn clean build -dStOp py $XTRA;
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
$PY $CWD/${PYGET}
mkn clean build test -O $XTRA
$PY test/asaga_sparse.py
