#!/bin/sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 CONF_FILE"
  exit 1
fi

for i in `seq 1 5`;
do
  python src/train.py --file=conf/$1 --name=$1.$i
done
