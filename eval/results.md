# Evaluated 20 queries over 8 documents (document-level ground truth, chunk-level retrieval, top-10)

## Overall

| Method                        | Vector backend             | P@1   | P@3   | Hit@3 | MRR   |
|-------------------------------|----------------------------|-------|-------|-------|-------|
| Keyword only (BM25)           | n/a                        | 0.900 | 0.583 | 0.950 | 0.933 |
| Vector only                   | TF-IDF                     | 0.850 | 0.617 | 0.950 | 0.902 |
| **Hybrid** (0.4 kw + 0.6 vec) | TF-IDF                     | 0.900 | 0.617 | 0.950 | 0.933 |
| Keyword only (BM25)           | n/a                        | 0.900 | 0.583 | 0.950 | 0.933 |
| Vector only                   | Dense (gemini-embedding-2) | 0.950 | 0.833 | 1.000 | 0.967 |
| **Hybrid** (0.4 kw + 0.6 vec) | Dense (gemini-embedding-2) | 1.000 | 0.783 | 1.000 | 1.000 |

### Hybrid lift (by MRR)

| Vector backend             | Best single method | MRR lift (abs) | MRR lift (rel) | W/T/L vs keyword | W/T/L vs vector |
|----------------------------|--------------------|----------------|----------------|------------------|-----------------|
| TF-IDF                     | keyword            | +0.000         | +0.0%          | 0/20/0           | 2/17/1          |
| Dense (gemini-embedding-2) | vector             | +0.033         | +3.4%          | 2/18/0           | 1/19/0          |

_W/T/L = queries where hybrid's reciprocal rank is higher / equal / lower._

### MRR by query type — TF-IDF vector backend

| Query type | n | Keyword | Vector | Hybrid |
|------------|---|---------|--------|--------|
| lexical    | 6 | 1.000   | 1.000  | 1.000  |
| mixed      | 5 | 1.000   | 0.900  | 1.000  |
| paraphrase | 9 | 0.852   | 0.837  | 0.852  |

### MRR by query type — Dense (gemini-embedding-2) vector backend

| Query type | n | Keyword | Vector | Hybrid |
|------------|---|---------|--------|--------|
| lexical    | 6 | 1.000   | 1.000  | 1.000  |
| mixed      | 5 | 1.000   | 0.867  | 1.000  |
| paraphrase | 9 | 0.852   | 1.000  | 1.000  |
