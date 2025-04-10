```shell
pip install pandas numpy faiss sentence_transformers scikit-learn
```

```shell
python3 script.py
```

* Define Queries in Test Cases
* Example Output: 

```
Query: Labourer Forest
Match: 9112/01 - Forest workers
Job Title: Labourer
Similarity: 0.946

Query: Labourer
Match: 9129/01 - Builder's labourers
Job Title: Labourer
Similarity: 0.715

Query: Shelf Stacker
Match: 9241/00 - Shelf fillers
Job Title: Stacker
Similarity: 0.882

Query: Shop Worker
Match: 9139/00 - Elementary process plant occupations n.e.c.
Job Title: Shopman
Similarity: 0.642

Query: Engineer
Match: }}}}/}} - }}}}/}}
Job Title: Engineer
Similarity: 0.699

Query: Warehouse worker
Match: 9252/00 - Warehouse operatives
Job Title: Warehouse operative
Similarity: 0.815

Query: Data scientist
Match: 2433/04 - Statistical data scientists
Job Title: Data scientist
Similarity: 0.853

Query: Hospital cleaner
Match: 9223/01 - Commercial cleaners
Job Title: Hospital cleaner
Similarity: 0.891

Query: Garbage collector
Match: 9225/02 - Refuse collectors
Job Title: Waste collector
Similarity: 0.808

Query: SDET
Match: 3131/03 - Quality assurance testers
Job Title: Qa tester
Similarity: 0.539

Query: Developer in Test
Match: 3131/03 - Quality assurance testers
Job Title: Qa tester
Similarity: 0.651

Query: Data Warehouse
Match: 2133/05 - IT business analysts
Job Title: Data warehouse analyst
Similarity: 0.746

Query: Software Developer
Match: 2134/03 - Software developers
Job Title: Developer
Similarity: 0.936

Query: Java Engineer
Match: 2134/03 - Software developers
Job Title: Java developer
Similarity: 0.883

Query: Java Developer
Match: 2134/03 - Software developers
Job Title: Java developer
Similarity: 0.938

Query: Javascript Developer
Match: 2134/03 - Software developers
Job Title: Javascript developer
Similarity: 0.914
```