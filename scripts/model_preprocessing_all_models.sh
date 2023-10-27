echo "Preprocessing $1 for model TRANSFORMER"
poetry run styler2_0 PREPROCESSING --project_dir "$1" --model TRANSFORMER

echo "Preprocessing $1 for model LSTM"
poetry run styler2_0 PREPROCESSING --project_dir "$1" --model LSTM
