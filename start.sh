#!/bin/bash

cd /app

if [[ $CLONE ]]
then
    echo "Cloning rp_handler.py and workflow.py"
    rm -rf rp_handler.py workflow.py
    curl https://raw.githubusercontent.com/sumbackup/multitalk_wan/refs/heads/main/rp_handler.py > rp_handler.py
    curl https://raw.githubusercontent.com/sumbackup/multitalk_wan/refs/heads/main/workflow.py > workflow.py
fi

echo "Starting rp_handler.py"
python rp_handler.py
