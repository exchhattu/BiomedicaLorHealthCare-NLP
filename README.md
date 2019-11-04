# Health Care/Biomedical NLP

### **Description**
The primary focus of the work is to recognize the entity from biomedical literature. 
This is a typical problem from the natural language processing domain and useful 
in a different spectrum of health care and biomedical research. 
This work has two goals - preprocess the data that includes data cleaning, 
preparation for transfer learning, and analysis and model development 
using deep learning network for prediction.

### **Requirements** 
* Python (3.6.0)
* Pandas (0.24.1)
* NumPy (1.16.0)
* Juypter (4.4.0)
* Matplotlib (3.0.2) 

### **Data**
Each of these data sources was published in scientific journals. 
The first two sources were curated manually and considered as gold standard data.
The pubtator provides curated data of 5.6 GB for a disease and is used to 
quantify the disease entity that exists in the gold standard data.

* [NCBIDisease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)
* [CDR](https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/)

* [pubtator](https://www.ncbi.nlm.nih.gov/research/pubtator/)

### Usage
```
$ python3 ./NLP_DNER.py -h
```

### Case study
Before using the commands provided below, download the NCBIDisease and CDR datasets 
from the linked described above. The program supports txt or XML file formats. 

```
$ python3 ./NLP_DNER.py -i ./data/NCBIdiseaseDataset/NCBI_corpus_gs.txt | tee ./NCBI_corpus_gs.out

$ python3 ./NLP_DNER.py -i ./data/CDR_Data/CDR.Corpus.v010516/CDR_gs_PubTator.txt | tee ./CDR_gs.out

$ cd ./notebook/
```

Notebooks contain analysis for multiplicity in disease named entity and quantification of disease entity 
useful for transfer learning 

### Disclamier
Opinions expressed are solely my own and do not express the views or opinions of my employer. 
The author assumes no responsibility or liability for any errors or omissions in the content of this site. 
The information contained in this site is provided on an “as is” basis with no guarantees of completeness, 
accuracy, usefulness or timeliness.
