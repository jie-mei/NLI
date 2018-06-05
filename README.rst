Attention Methods for Pairwised Document Similarity Analysis
============================================================

Comparison of Attention Methods for Document Similarity Analysis.


Usage
-----

.. role:: bash(code)
   :language: bash


.. code:: bash

  python src/eval.py --file=conf/SNLI.GloVeNorm.attentive.yaml --name=NAME

..

The optional `name` flag applys a customize name. Otherwise, the generated files
will be stored at `build/SNLI.GloVeNorm.attentive` and override the old files if
running multiple times.


Requirements
------------

* python (>= 3.5)
* gensim
* loggy
* nltk
* scikit-learn
* tensorflow
* tqdm
