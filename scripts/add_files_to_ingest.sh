TYPE=$1

python src/ingest/add_files.py --type "$TYPE" --files "${@:2}"