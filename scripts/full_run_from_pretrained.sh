input_dir=$1
output_dir=$2
pre_training_dir=$3

echo "Mining violations for $input_dir"
source scripts/mine_violations.sh "$input_dir" "$output_dir"

echo "Generate violations for $input_dir"
source scripts/preprocess_all_protocols.sh "$input_dir" "$output_dir"

echo "Preprocessing for all models"
echo "Preprocessing $output_dir for model TRANSFORMER"
poetry run styler2_0 PREPROCESSING --project_dir "$output_dir" --model TRANSFORMER --src_vocab "$pre_training_dir"/transformer/src_vocab.txt --trg_vocab "$pre_training_dir"/transformer/trg_vocab.txt

echo "Preprocessing $output_dir for model LSTM"
poetry run styler2_0 PREPROCESSING --project_dir "$output_dir" --model LSTM --src_vocab "$pre_training_dir"/lstm/src_vocab.txt --trg_vocab "$pre_training_dir"/lstm/trg_vocab.txt

echo "Preprocessing $output_dir for model ANN"
poetry run styler2_0 PREPROCESSING --project_dir "$output_dir" --model ANN --src_vocab "$pre_training_dir"/ann/src_vocab.txt --trg_vocab "$pre_training_dir"/ann/trg_vocab.txt

echo "Preprocessing $output_dir for model NGRAM"
poetry run styler2_0 PREPROCESSING --project_dir "$output_dir" --model NGRAM --src_vocab "$pre_training_dir"/ngram/src_vocab.txt --trg_vocab "$pre_training_dir"/ngram/trg_vocab.txt

echo "Training all models"
source scripts/train_all_models.sh "$output_dir"/model_data
echo "Train LSTM on $1"
poetry run styler2_0 TRAIN --model LSTM --epochs 100 --path "$1" --lr 1e-5 --from_checkpoint "$pre_training_dir"/lstm/checkpoints/lstm.pt

echo "Train TRANSFORMER on $1"
poetry run styler2_0 TRAIN --model TRANSFORMER --epochs 100 --path "$1" --lr 1e-5 --from_checkpoint "$pre_training_dir"/transformer/checkpoints/lstm.pt

echo "Train ANN on $1"
poetry run styler2_0 TRAIN --model ANN --epochs 100 --path "$1" --lr 1e-5 --from_checkpoint "$pre_training_dir"/ann/checkpoints/lstm.pt

echo "Train NGRAM on $1"
poetry run styler2_0 TRAIN --model NGRAM --epochs 100 --path "$1" --lr 1e-5 --from_checkpoint "$pre_training_dir"/ngram/checkpoints/lstm.pt


echo "Evaluating all models"
source scripts/eval_all_models_on_all_protocols.sh "$output_dir"/mined_violations "$output_dir"
