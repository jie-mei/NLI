PYTHON=python3
PROJECT:=$(shell basename $(shell pwd))

.PHONY: download train test clean


train:
	$(PYTHON) src/train.py --file config/MSRP.decompose.yaml

test:
	$(PYTHON) src/test.py --file config/MSRP.decompose.yaml

# Use conda to manage the dependency if applicable
download: data/ASS data/MSRP data/WikiQA data/SemEval
	hash conda > /dev/null && while read requirement; do conda install --yes "$$requirement"; done < requirements.txt > /dev/null 2>&1
	$(PYTHON) -m pip install -qr requirements.txt
	$(PYTHON) -m nltk.downloader punkt
	$(PYTHON) -m spacy download en_vectors_web_lg

data/ASS:
	scripts/download_ASS     data

data/MSRP:
	scripts/dwnload_MSRP     data

data/WikiQA:
	scripts/download_WikiQA  data

data/SemEval:
	scripts/download_SemEval data

clean:
	rm -rf build

#-------------------------------------------------------------------------------

.PHONY: deploy tensorboard

# Visualize statistics using tensorboard
tensorboard: ~/Mount/bigdata1/$(PROJECT)
	tensorboard --logdir="~/Mount/bigdata-gpu1/$(PROJECT)/build/models"

~/Mount/bigdata1/$(PROJECT):
	sshfs jmei@bigdata-gpu1.research.cs.dal.ca:/home/jmei \
			~/Mount/bigdata-gpu1/
