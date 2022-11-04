#!/bin/bash

# Convert all .ipynb files in the repo to .py files

#source ~/.bashrc
#conda init bash
#conda activate ENV_ML

if [ "$1" == "--all" ]; then
	SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
	for notebook in $(find "$SCRIPT_DIR" | grep ".ipynb$"); do
		jupyter nbconvert --to script $notebook
	done
elif [ "$1" == "" ]; then
	echo "Usage: jup_to_py.sh [nb.ipynb | --all]"
	exit
else
	jupyter nbconvert --to script $1
fi
