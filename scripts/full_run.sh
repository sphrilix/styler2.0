input_dir=$1
output_dir=$2

echo "Mining violations for $input_dir"
source scripts/mine_violations.sh "$input_dir" "$output_dir"

echo "Generate violations for $input_dir"
source scripts/preprocess_all_protocols.sh "$input_dir" "$output_dir"

echo "Preprocessing for all models"
source scripts/model_preprocessing_all_models.sh "$output_dir"

echo "Training all models"
source scripts/train_all_models.sh "$output_dir"/model_data

echo "Evaluating all models"
source scripts/eval_all_models_on_all_protocols.sh "$output_dir"/mined_violations
