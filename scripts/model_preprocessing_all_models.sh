echo "Preprocessing $1 for model TRANSFORMER"
poetry run styler2_0 PREPROCESSING --project_dir "$1" --model TRANSFORMER

echo "Preprocessing $1 for model LSTM"
poetry run styler2_0 PREPROCESSING --project_dir "$1" --model LSTM

echo "Preprocessing $1 for model ANN"
poetry run styler2_0 PREPROCESSING --project_dir "$1" --model ANN

echo "Preprocessing $1 for model N_GRAM"
poetry run styler2_0 PREPROCESSING --project_dir "$1" --model N_GRAM
