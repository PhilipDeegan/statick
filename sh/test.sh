#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd $CWD/.. && CWD=$PWD
PY="${PY:-python3}";
which $PY; $PY -V

export PYTHONPATH="$CWD/tick:$CWD"
export MKN_LIB_LINK_LIB=1
PYGET="get_data.py";
function finish { cd $CWD; }; trap finish EXIT;
ARF="adult.features.cereal"; ARL="adult.labels.cereal";
URF="url.features.cereal"; URL="url.labels.cereal";
[ ! -d "url" ] && git clone https://github.com/PhilipDeegan/statick_data --depth 1 -b master url && \
                  (cd url && ./join.sh url_svmlight.tar.gz && rm url_svmlight.tar.gz.part.*)
mkn -v; mkn clean build -dStO 2 -p py $XTRA;
cat > $CWD/${PYGET} << EOL
import statick
from tick.dataset.download_helper import fetch_tick_dataset
from tick.dataset.fetch_url_dataset import load_url_dataset_day, dataset_path as url_dataset_path
load_url_dataset_day
adult_train_set = fetch_tick_dataset('binary/adult/adult.trn.bz2')
url_10_days = load_url_dataset_day(url_dataset_path, range(10))
statick.save_double_sparse2d(adult_train_set[0], "${ARF}")
statick.save_double_array(adult_train_set[1], "${ARL}")
statick.save_double_sparse2d(url_10_days[0], "${URF}")
statick.save_double_array(url_10_days[1], "${URL}")
EOL
$PY $CWD/${PYGET}
mkn clean build test -O $XTRA
$PY test/asaga_sparse.py
