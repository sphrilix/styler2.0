echo "Evaluating LSTM on $1"
poetry run styler2_0 EVAL --model LSTM --mined_violations_dir "$1" --project_dir "$2"

echo "Evaluating TRANSFORMER on $1"
poetry run styler2_0 EVAL --model TRANSFORMER --mined_violations_dir "$1" --project_dir "$2"

echo "Evaluating ANN on $1"
poetry run styler2_0 EVAL --model ANN --mined_violations_dir "$1" --project_dir "$2"

echo "Evaluating NGRAM on $1"
poetry run styler2_0 EVAL --model NGRAM --mined_violations_dir "$1" --project_dir "$2"
