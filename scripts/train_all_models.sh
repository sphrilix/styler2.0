echo "Train LSTM on $1"
poetry run styler2_0 TRAIN --model LSTM --epochs 100 --path "$1"

echo "Train TRANSFORMER on $1"
poetry run styler2_0 TRAIN --model TRANSFORMER --epochs 100 --path "$1"

echo "Train ANN on $1"
poetry run styler2_0 TRAIN --model ANN --epochs 100 --path "$1"

echo "Train NGRAM on $1"
poetry run styler2_0 TRAIN --model NGRAM --epochs 100 --path "$1"
