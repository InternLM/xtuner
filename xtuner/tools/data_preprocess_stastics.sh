ROOT=/path/to/xtuner
export PYTHONPATH=$ROOT:$PYTHONPATH

export OMP_NUM_THREADS=1

# $1: internvl_sft_1.2M.json, data format as https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#meta-file
# $2: the folder of the results of token stastics which is absolute path
# $3: pack_internvl_sft_1.2M.json, results file
python data_preprocess_stastics.py --json_file $1 --token_lengths_path $2 --output_path $3 2>&1 | tee -a log_statistics.txt
