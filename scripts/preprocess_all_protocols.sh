echo "Preprocessing $1 with protocol RANDOM"
poetry run styler2_0 GENERATE_VIOLATIONS --source "$1" --save "$2" --protocol RANDOM --n 100 --delta 36000

echo "Preprocessing $1 with protocol THREE_GRAM"
poetry run styler2_0 GENERATE_VIOLATIONS --source "$1" --save "$2" --protocol THREE_GRAM --n 100 --delta 36000
