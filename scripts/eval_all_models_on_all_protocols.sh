random=$1/../model_data/random
three_gram=$1/../model_data/three_gram

echo "Evaluating LSTM on $1"
poetry run styler2_0 EVAL --model LSTM --mined_violations_dir "$1" --model_data_dirs "$random"/lstm "$three_gram"/lstm --checkpoints "$random"/lstm/checkpoints/lstm.pt "$three_gram"/lstm/checkpoints/lstm.pt

echo "Evaluating TRANSFORMER on $1"
poetry run styler2_0 EVAL --model TRANSFORMER --mined_violations_dir "$1" --model_data_dirs "$random"/transformer "$three_gram"/transformer --checkpoints "$random"/transformer/checkpoints/transformer.pt "$three_gram"/transformer/checkpoints/transformer.pt
