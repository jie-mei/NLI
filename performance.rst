
SemEval Short Text Similarity (STS)
-----------------------------------

:Original: `SemEval-2016 Task 1`_
:Source:   `brmson/dataset-sts`_

Microsoft Research Paraphrase Corpus (MSRP)
-------------------------------------------

:Original: `Microsoft Research Paraphrase Corpus`_
:Source:   `brmson/dataset-sts`_

A corpus for sentence paraphrase, paraphrase identification, paraphrase recognition. Evaluated Accuracy/F results are reported on the ACL `Paraphrase Identification (state-of-the-art)`_ and MAP/MRR results are reported on `(Yin et al, 2016)`_.

+-----------------------+----------------------+------------+------------+----------+----+
|  Algorithm            | Reference            |   MAP      |   MRR      | Accuracy | F  |
+===========+===========+======================+============+============+==========+====+
|  BCNN     |  [1conv]  | `(Yin et al, 2016)`_ |   0.6629   |   0.6813   |          |    |
+           +-----------+                      +------------+------------+----------+----+
|           |  [2conv]  |                      |   0.6593   |   0.6738   |          |    |
+-----------+-----------+                      +------------+------------+----------+----+
|  ABCNN-1  |  [1conv]  |                      |   0.6810   |   0.6979   |          |    |
+           +-----------+                      +------------+------------+----------+----+
|           |  [2conv]  |                      |   0.6855   |   0.7023   |          |    |
+-----------+-----------+                      +------------+------------+----------+----+
|  ABCNN-2  |  [1conv]  |                      |   0.6885   |   0.7054   |          |    |
+           +-----------+                      +------------+------------+----------+----+
|           |  [2conv]  |                      |   0.6879   |   0.7068   |          |    |
+-----------+-----------+                      +------------+------------+----------+----+
|  ABCNN-3  |  [1conv]  |                      |   0.6914   | **0.7127** |          |    |
+           +-----------+                      +------------+------------+----------+----+
|           |  [2conv]  |                      | **0.6921** |   0.7108   |          |    |
+-----------+-----------+----------------------+------------+------------+----------+----+


Answer Sentence Selection (ASS)
-------------------------------

:Source: `brmson/dataset-sts`_
:Report: `Question Answering (state-of-the-art)`_
:TODO:   Clean or All dataset?


Microsoft Research WikiQA Corpus (wikiQA)
-----------------------------------------

:Source: `Microsoft Research WikiQA Corpus`_
:Abstract:
  The WikiQA corpus is a new publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering. In order to reflect the true information need of general users, we used Bing query logs as the question source. Each question is linked to a Wikipedia page that potentially has the answer. Because the summary section of a Wikipedia page provides the basic and usually most important information about the topic, we used sentences in this section as the candidate answers. With the help of crowdsourcing, we included 3,047 questions and 29,258 sentences in the dataset, where 1,473 sentences were labeled as answer sentences to their corresponding questions. More detail of this corpus can be found in our EMNLP-2015 paper, "WikiQA: A Challenge Dataset for Open-Domain Question Answering" [Yang et al. 2015]. In addition, this download also includes the experimental results in the paper, an evaluation script for judging the "answer triggering" task, as well as the answer phrases labeled by the authors of the paper. 
:Report: `(Yin et al, 2016)`_
:Format: `Tokenized question`, `Tokenized candidate answer sentence`, `Label`


+-----------------------+----------------------+----------+----------+
|  Algorithm            | Reference            | Accuracy |   F1     |
+===========+===========+======================+==========+==========+
|  BCNN     |  [1conv]  | `(Yin et al, 2016)`_ |   78.1   |   84.1   |
+           +-----------+                      +----------+----------+
|           |  [2conv]  |                      |   78.3   |   84.3   |
+-----------+-----------+                      +----------+----------+
|  ABCNN-1  |  [1conv]  |                      |   78.5   |   84.5   |
+           +-----------+                      +----------+----------+
|           |  [2conv]  |                      |   78.5   |   84.6   |
+-----------+-----------+                      +----------+----------+
|  ABCNN-2  |  [1conv]  |                      |   78.6   |   84.7   |
+           +-----------+                      +----------+----------+
|           |  [2conv]  |                      |   78.8   |   84.7   |
+-----------+-----------+                      +----------+----------+
|  ABCNN-3  |  [1conv]  |                      |   78.8   | **84.8** |
+           +-----------+                      +----------+----------+
|           |  [2conv]  |                      | **78.9** | **84.8** |
+-----------+-----------+----------------------+----------+----------+


.. _Microsoft Research Paraphrase Corpus: https://www.microsoft.com/en-us/download/details.aspx?id=52398
.. _Microsoft Research WikiQA Corpus: https://www.microsoft.com/en-us/download/details.aspx?id=52398
.. _Paraphrase Identification (state-of-the-art): https://aclweb.org/aclwiki/Paraphrase_Identification_(State_of_the_art)
.. _Question Answering (state-of-the-art): https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)
.. _SemEval-2016 Task 1: http://alt.qcri.org/semeval2016/task1/index.php?id=data-and-tools
.. _(Yin et al, 2016): http://aclweb.org/anthology/Q/Q16/Q16-1019.pdf
.. _brmson/dataset-sts: https://github.com/brmson/dataset-sts
