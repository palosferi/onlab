#!/bin/bash

sudo pkill -f "chrome"
sudo pkill -x "tcpdump"

cd /home/palos/tor_wf_project

source venv/bin/activate

python scripts/collector.py >> logs/collection.log 2>&1
