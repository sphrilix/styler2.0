echo "Train LSTM on $1"
poetry run styler2_0 TRAIN --model LSTM --epochs 1 --path "$1"

echo "Train TRANSFORMER on $1"
poetry run styler2_0 TRAIN --model TRANSFORMER --epochs 1 --path "$1"
