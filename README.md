# Multi-Document Summarization for Russian
### Authors: Alexandra Konovalova, Alina Tillabaeva, Egor Plotnikov, Elizaveta Klykova
## Data
Summaries of 944 chapters in 67 Russian classical books.  
[Dataset at huggingface.co](https://huggingface.co/datasets/c00k1ez/summarization)
## Experiments
Comparison of TextRank, Hierarchical and pre-trained multilingual mBART and mT5 algorithms.
## Summary Evaluation
We propose a summarization evaluation metric based on several existing algorithms. Our metric is composed of the following steps:
1. Keyword extraction
2. Calculation of BERTScore between keywords from the chapter and those from the summary
3. Named Entity extraction (the intersection of NE in the chapter and the summary is considered)
4. Calculation of ROUGE-L
5. Adjustment for chapther length
## Gold Standard
Various summaries of 7 chapters in different books were collected and evaluated by experts on a scale from 1 to 5 (1 = very bad, 5 = excellent). The chapters were all different in word count to reduce possible length bias of the metrics. 102 summaries (\~15 per chapter) were collected.

The dataset file **gold_standard.xlsx** contains the following fields: book_id, book_title, chapter_id, chapter_title, chapter_summary, summary_source, length_score (length of summary / length of chapter), human_score, is_best, comments, length_summary (words).
