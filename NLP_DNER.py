'''
Written: Rojan Shrestha PhD
Mon Sep  9 18:09:11 2019
'''
"""
Opinions expressed are solely my own and do not express the views or opinions of my employer. 
The author assumes no responsibility or liability for any errors or omissions in the content of this site. 
The information contained in this site is provided on an “as is” basis with no guarantees of completeness, 
accuracy, usefulness or timeliness.
"""
import re, sys, os, shutil
import argparse

import numpy as np

from gensim.models.keyedvectors import KeyedVectors

import nltk
from nltk import tokenize, RegexpTokenizer 
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags

import spacy

import xml.etree.ElementTree as XMLtree 

from model import Model

class NcbiDiseaseCorpus:

  class Annotation:

    def __init__(self, st_line):
      ar_parts = st_line.split("\t")
      self._start_pos   = ar_parts[1] 
      self._end_pos     = ar_parts[2] 
      self._de_name     = ar_parts[3] 
      if len(ar_parts) > 4: 
        self._de_type     = ar_parts[4] 
        self._de_class    = ar_parts[5] 
      else:
        self._de_type     = ""  
        self._de_class    = "" 

      # self.verbose()

    def verbose(self):
      print("[VERBOSE] {} {} {} {} {}".\
            format(self._start_pos, self._end_pos, 
                   self._de_name, self._de_type, self._de_class))
    def get_bert_format(self):
      """ annotation format in bert.  """
      return "Disease %s %s" %(self._start_pos, self._end_pos) 

    def get_disease_name(self):
      return self._de_name

    def get_disease_class(self):
      return self._de_class
      
  class Corpus:
    # def __init__(self, abstract, title, pubmed_id, annotations):
    def __init__(self, pubmed_id, abstract, title, annotations):
      self._pubmed_id   = pubmed_id
      self._abstract    = abstract 
      self._title       = title 
      # self.verbose()
      self._anns        = annotations 
      self._method = "nltk"

      self._go_anns = [] # secondary annotation
      # if self._method == "spacy":
      self._oj_sp = spacy.load('en_core_web_sm')
      self._is_valid_silver = False

    def update_silver_validity(self, value=True):
      self._is_valid_silver = value

    def append_annotation(self, st_disease_info):
      self._go_anns.append(st_disease_info)

    def get_abstract(self):
      return self._abstract

    def verbose(self):
      print("[VERBOSE] Title {}".format(self._title))
      print("[VERBOSE] Abstract {}".format(self._abstract))

    def parse_abstract(self):
      # if self._method == "spacy":
      doc_abstract = self._oj_sp(self._abstract) 
      self._ts_abs_sent_tokens = [ sentence for sentence in doc_abstract.sents ]
      self._ts_abs_word_tokens = [ sentence.text for sentence in doc_abstract ] 
      # else: # default 
      #   self._ts_abs_sent_tokens = sent_tokenize(self._abstract)
      #   self._ts_abs_word_tokens = word_tokenize(self._abstract)

    def get_num_words_sentences(self):
      return (len(self._ts_abs_word_tokens), len(self._ts_abs_sent_tokens))

    def tag_bio(self): 
      #print(ne_chunk(pos_tag(self._ts_abs_word_tokens))) 
      iob_tagged = tree2conlltags(ne_chunk(pos_tag(self._ts_abs_word_tokens)))
      print(iob_tagged)

    def write_annotation_bert_format(self, ts_anns, dir_path=os.getcwd()):
      """
        Aims: write the annotations into a file.

        Params:
          ts_anns: list of disease annotations
          dir_path: path to directory
      """
      count = 1
      st_annotation = "%s.ana" %self._pubmed_id
      st_annotation = os.path.join(dir_path, st_annotation)
      with open(st_annotation, "w") as oj_writer:
        for oj_ann in self._anns:
          tag = "T%d" %count 
          pos = oj_ann.get_bert_format()
          oj_writer.write(tag+"\t"+pos+"\t"+oj_ann._de_name+"\n")
          count +=1

    def create_bert_format(self, dir_path=os.getcwd()):
      """
        Aims: write annotation file in bert format. While using transfer learning, there
              are two annotations - gold and silver annotation. Gold is manually curated
              whereas silver is curated using automation. 
        Params:
              dir_path: path to directory where files are saved.
      """
      if (self._anns or self._go_anns): 
        st_content="%s.txt" %self._pubmed_id
        st_content = os.path.join(dir_path, st_content)
        with open(st_content, "w") as oj_writer:
          oj_writer.write(self._abstract)
   
        if self._anns and self._go_anns:
          self.write_annotation_bert_format(self._go_anns, dir_path)
        elif self._anns:
          self.write_annotation_bert_format(self._anns, dir_path)


    def get_disease_names(self):
      return [ oj_ann.get_disease_name().lower() for oj_ann in self._anns]

    def get_disease_classes(self):
      return [ oj_ann.get_disease_class() for oj_ann in self._anns]

  def __init__(self):
    self._dc_gs_diseases = {} # keep tokenize disease name based on space
    self._ts_gs_diseases = [] # unique disease name without word based tokenization

  def get_statistics(self, dump_pubmed_id=False, dump_disease_name=False, tag="gold"):
    di_corpus = {}
    if tag == "gold":
      di_corpus = self._di_pubmeds.copy()
    elif tag == "silver":
      di_corpus = self._di_si_pubmed.copy()

    ts_disease_names = []
    print("[INFO]: # of abstract: {}".format(len(di_corpus.keys())))
    in_g_word = 0
    in_g_sent = 0

    di_disease_class_names  = {}
    di_disease_name_maps    = {}
    di_disease_class_names_rev = {}
    for k, oj_corpus in di_corpus.items():
      oj_corpus.parse_abstract()
      in_num_word, in_num_sent = oj_corpus.get_num_words_sentences()
      in_g_word += in_num_word
      in_g_sent += in_num_sent

      # Work with disease name
      ts_disease_name_temp  = oj_corpus.get_disease_names()

      # make mapping between disease id and name 
      ts_disease_classes    = oj_corpus.get_disease_classes()
      for st_id, st_name in zip(ts_disease_classes, ts_disease_name_temp):
        di_disease_class_names[st_id] = di_disease_class_names.get(st_id, [])
        di_disease_class_names[st_id].append(st_name)

        di_disease_class_names_rev[st_name] = st_id 

      ts_disease_names += oj_corpus.get_disease_names()
      
    print("[INFO]: # of sentences and words: {} and {}".format(in_g_word, in_g_sent))
    print("[INFO]: # of annotated disease name: {}".format(len(ts_disease_names)))

    ar_unique_dnames, ar_counts = np.unique(ts_disease_names, return_counts=True)
    print("[INFO]: # of uniquely annotated disease name: {}".format(len(ar_unique_dnames)))

    for st_did, ts_list in  di_disease_class_names.items(): 
      print("WordsPerDisease: %s,%d,%s" %(st_did, len(np.unique(ts_list)), ",".join(np.unique(ts_list))))

    # Count uniqueness
    di_disease_unique = {}
    for st_dname, in_dname_count in zip(ar_unique_dnames, ar_counts): 
      if not st_dname in di_disease_class_names_rev.keys():  
        print("[WARNING]: NOT FOUND %" %st_dname)
        continue

      if not di_disease_class_names_rev[st_dname] in di_disease_unique.keys(): 
        di_disease_unique[di_disease_class_names_rev[st_dname]] = in_dname_count
      else:
        di_disease_unique[di_disease_class_names_rev[st_dname]] += in_dname_count

    for st_di_code, in_di_count in di_disease_unique.items():
      print("UniqueDiseaseCount: %s,%d" %(st_di_code, in_di_count))

    if dump_disease_name: self.write_to_file("%s_UniqueDName.txt" %tag.title(), ar_unique_dnames)
    ar_pubmed_ids = np.array(list(di_corpus.keys()))
    if dump_pubmed_id: np.savetxt("%s_UniquePubmedID.txt" %tag.title(), ar_pubmed_ids, fmt="%s")

  def write_to_file(self, path_to_file, ar_to_write):
    with open(path_to_file, "w") as oj_writer:
      for st_line in ar_to_write:
        oj_writer.write(st_line+"\n")
      
  def tag_bio_g(self):
    for k, v in self._di_pubmeds.items():
      oj_corpus = self._di_pubmeds[k] 
      oj_corpus.tag_bio()

  def convert_in_bert_format(self):
    """
      Aims: write bert format files - .txt and .ann for each pubmed.
      In .txt file, it only contains abstract where .ann is annotated list of disease 
      with their positions.
    """
    count = 0 
    for k, v in self._di_pubmeds.items():
      count += 1
      oj_corpus = self._di_pubmeds[k] 
      oj_corpus.create_bert_format()
    print("[INFO]: write {} .txt/.ann files...".format(count))

  def get_pubmed_ids(self):
    return np.array(list(self._di_pubmeds.keys()))

  def parse_ncbi_disease_corpus(self, path_to_data="", write_output=False):
    self._di_pubmeds = {}
    pubmed_id=""; jtitle=""; jabstract=""; ts_anns = []
    if not os.path.exists(path_to_data):
      print("[FATAL]: file does not exist (%s)".format(path_to_data))
      sys.exit(0)

    with open(path_to_data, "r") as oj_reader:
      ar_lines = oj_reader.read().split("\n")
      for st_line in ar_lines:
        if not st_line: # this is next document 
          if not pubmed_id: continue  
          self._di_pubmeds[pubmed_id] = self.Corpus(pubmed_id, jabstract, jtitle, ts_anns)
          pubmed_id=""; jtitle=""; jabstract=""; ts_anns = []
          continue
        if "|t|" in st_line: # title 
          st_parts      = st_line.split("|t|")
          pubmed_id = st_parts[0] # pubmed id  
          jtitle    = st_parts[1] # title  
          
        if "|a|" in st_line: # abstract 
          st_parts          = st_line.split("|a|")
          assert pubmed_id == st_parts[0] # pubmed id is not matching  
          jabstract    = st_parts[1] # title  

        if st_line.startswith(pubmed_id) and not ("|t|" in st_line or "|a|" in st_line):
          if not pubmed_id in st_line: 
            print("[FATAL]: pubmed id {} mismatch!".format(pubmed_id))
          else:
            ts_anns.append(self.Annotation(st_line))

  def parseXMLPubTator(self, path_xml_file, write_bart_format=False):
    """
    Aims: 
      parse XML data and extract few information based on their attribute id

    params:
      path_xml_file: path to XML file
      write_bart_format: write into bart file format, if value is True
    """

    self._di_si_pubmed= {} # silver standard data
    oj_tree = XMLtree.parse(path_xml_file)
    oj_root = oj_tree.getroot()

    pubmed_id=""; jtitle=""; jabstract=""; ts_anns = []
    # for oj_docs in oj_root.findall('document'):
    for oj_docs in oj_root.findall('PubmedArticle'):
      # pubmed_id = oj_docs.find("id").text

      pubmed_id = oj_docs.find("PMID").text
      print("Coding: pubmed id ", pubmed_id)

      st_text = "" # title and abstract
      st_annotation = ""
      in_mod_id = 0 
      for oj_passage in oj_docs.findall('passage'):
        oj_infon = oj_passage.find('infon') 
        jabstract += oj_passage.find("text").text 
        
        for oj_annotation in oj_passage.findall('annotation'):
          for oj_nested_infon in oj_annotation.findall('infon'):
            if oj_nested_infon.get("key")=="type" and oj_nested_infon.text == 'Disease': 
              st_annotation = oj_nested_infon.text
              in_spos = int(oj_annotation.find('location').get('offset')) 
              in_epos = in_spos + int(oj_annotation.find('location').get('length')) 
              st_annotation += "\t" + str(in_spos) + "\t" + str(in_epos) 
              st_annotation += "\t" + oj_annotation.find('text').text.title() 
              ts_anns.append(self.Annotation(st_annotation))
              in_mod_id += 1 
              st_annotation = "T" + str(in_mod_id) + "\t" + st_annotation + "\n"

        if not pubmed_id: break 
        self._di_si_pubmed[pubmed_id] = self.Corpus(pubmed_id, jabstract, jtitle, ts_anns)
        pubmed_id=""; jtitle=""; jabstract=""; ts_anns = []
    
    if write_bart_format:
      with open("%s.txt" %pubmed_id, "w") as oj_txt_writer: 
        oj_txt_writer.write(jabstract)

      with open("%s.ann" %pubmed_id, "w") as oj_ann_writer: 
        oj_ann_writer.write(st_annotation)

  def tokenize_gold_disease_name(self):
    """
      Aim: 
        Tokenize the gold standard disease name. For enhancing the search, 
        # of space was also monitored. 
    """

    pat = re.compile("[^\S]+") # pattern for space
    for k, oj_corpus in self._di_pubmeds.items(): # gold standard
      self._ts_gs_diseases += oj_corpus.get_disease_names()

    # Each disease name from gold standard data is tokenized in order to 
    # enhance the search in silver standard corpus 
    for st in self._ts_gs_diseases:
      if not st: continue
      in_num_sp = len(re.findall(pat, st))
      self._dc_gs_diseases[in_num_sp] = self._dc_gs_diseases.get(in_num_sp, [])
      self._dc_gs_diseases[in_num_sp].append(st) 

    # summary:
    for k, v in self._dc_gs_diseases.items():
      print("[INFO]: disease name w/ {} space(s) {}".format(k, len(list(v))))


  def compare_diseases_annotation(self):
    """
      Aims: compare silver standard and gold standard disease annotation
    """

    ts_si_diseases = []
    for k, oj_corpus in self._di_si_pubmed.items(): # silver standard data
      ts_si_diseases += oj_corpus.get_disease_names()

    se_common = set(ts_si_diseases).intersection(set(self._ts_gs_diseases))
    print("[INFO]: # of shared disease entity annotation: {}".format(len(se_common)))

    se_difference = set(np.unique(ts_si_diseases)).difference(set(np.unique(self._ts_gs_diseases)))
    print("[INFO]: # of extra disease entity annotation in silver standard dataset: {}".format(len(se_difference)))


  def compare_pubmed_ids(self):
    """
     Aims: compare silver standard corpus with gold standard 
    """
    ts_gs_pubmedids     = set(np.unique(list(self._di_pubmeds.keys())))
    ts_si_pubmedids     = set(np.unique(list(self._di_si_pubmed.keys())))

    ts_intersection = ts_gs_pubmedids.intersection(ts_si_pubmedids)
    print("[INFO]: # of gold standard abstracts: %d" %len(ts_gs_pubmedids))
    print("[INFO]: # of silver standard abstracts: %d" %len(ts_si_pubmedids))
    print("[INFO]: # of matches: %d" %len(ts_intersection))


  def find_DE_in_corpus(self, write_bart_format=False): 
    """
      Aims: 
        find a gold standard disease name in silver standard corpus.
        If gold standard disease name is in the silver corpus,
        it will create new annotation files. Otherwise, these pubmeds
        are ignored for downstream training and testing purposes.
      
      Param:
        write_bart_format: indicates whether to write or not. If fwrite=True and 
                           gold standard entity is found in silver corpus, it write 
                           the .txt and .ann files. Files are in bart format. 
    """
    # get gold standard corpus pubmed ids
    ts_gs_pubmedids     = np.unique(list(self._di_pubmeds.keys()))

    ts_common_pubmedids = []
    pat = re.compile("[^\S]+") # pattern for space
    for k, oj_corpus in self._di_si_pubmed.items(): # silver standard data
      if k in ts_gs_pubmedids: # if exists in gold standard, ignore it. 
        ts_common_pubmedids.append(k)
        continue

      st_abstract = oj_corpus.get_abstract()

      # make hash table
      di_word_hash = {} # word, their position
      in_spos = 0
      for word in pat.finditer(st_abstract):
        tu_start = word.span()
        in_diff = int(tu_start[1]) - int(tu_start[0])
        in_epos = int(tu_start[0])
        if not in_diff == 1: print("[WARNING]: more space length is found.")
        st_key_word = st_abstract[in_spos:in_epos]
        di_word_hash[st_key_word] = di_word_hash.get(st_key_word, [])
        di_word_hash[st_key_word].append((in_spos, in_epos))
        in_spos = int(tu_start[1])
      
      # Last word in a paragraph is not covered 
      st_key_word = st_abstract[in_spos:]
      di_word_hash[st_key_word] = di_word_hash.get(st_key_word, [])
      di_word_hash[st_key_word].append((in_spos, len(st_abstract)))

      # search gold standard disease name 
      is_found = False
      for in_sp in self._dc_gs_diseases.keys():  
        for st_di in self._dc_gs_diseases[in_sp]:
          ts_spos = []
          ts_epos = []
          for word in st_di.split(" "):
            if not word in di_word_hash: break 
            tuples = di_word_hash[word] 
            if not ts_spos and not ts_epos:
              for tuple in tuples:
                ts_spos.append(tuple[0])
                ts_epos.append(tuple[1])
            else:
              for i, tuple in enumerate(tuples):
                if (ts_epos[i] - tuple[0]) == 1: 
                  ts_epos[i] = tuple[1]
                else:
                  break
          if ts_spos and ts_epos:
            is_found = True
            st_disease_info = "%s %s %s" %(st_di, ts_spos, ts_epos)
            oj_corpus.append_annotation(st_disease_info)
      if is_found: # update only new disease entity is found 
        oj_corpus.update_silver_validity(True)
        self._di_si_pubmed[k] = oj_corpus
        if write_bart_format:
          with open("%s.txt" %k, "w") as oj_txt_writer: 
            oj_txt_writer.write(jabstract)

          with open("%s.ann" %pubmed_id, "w") as oj_ann_writer: 
            oj_ann_writer.write(st_annotation)

    if len(ts_gs_pubmedids) > 0:
      print("[INFO]: share pubmed ids: {}".format(len(ts_gs_pubmedids)))

  def write_valid_silver_corpus_annotation(self, dir_path=os.getcwd()):
    """
      Aims: write abstract and annotation in brat format
      Params: dir_path - path to directory
    """

    in_pubmed_count = len(self._di_si_pubmed.items())
    if in_pubmed_count==0: return
    print("[INFO]: valid silver corpus {}".format(len(self._di_si_pubmed.items())))
    # create a directory that can easily distinguish silver from gold
    path = os.path.join(dir_path, "silver_standard_data")
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)

    for k, v in self._di_si_pubmed.items():
      if not os.path.exists(os.path.join(dir_path, "%s.txt" %k)):
        v.create_bert_format(dir_path) 

  def convert_bin_to_glove(self, path_input_binary_file):
    """
      Aims: binary C format file is converted to plain text format that is 
            accepted by NeuroNER. The output will be generated in current
            working directory.
      Params: 
        path_input_binary_file: input for binary file  
    """
    if not os.path.exists(path_input_binary_file):
      print("[INFO]: {} does not exist".format(path_input_binary_file))
      sys.exit(0)
    
    # load word vectors provided in C binary format
    oj_word_vectors = KeyedVectors.load_word2vec_format(path_input_binary_file, binary=True)
    ts_vocabs       = oj_word_vectors.vocab
    # write vocabularies into a file in glove format 
    st_txt_file = "plain_word_vectors.txt"

    with open(st_txt_file, 'w+') as oj_f:
      for st_word in ts_vocabs:
        ts_word_vector = oj_word_vectors[st_word]
        oj_f.write("%s %s\n" %(st_word, " ".join(str(v) for v in ts_word_vector)))
            
    print('[INFO]: Converted binary file into plain text and saved...')
    print('{}'.format(st_txt_file))



if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='NLP_DNER.py', description='Name Entity Recoginition')
  parser.add_argument('-i','--path_to_input', required=True, help='Path to input file')
  parser.add_argument('-a','--path_to_binary', required=False, help='Path to binary embedding file')
  parser.add_argument('-u','--unit_test', required=False, default=False, action='store_true', \
                      help='Unit test on')
  parser.add_argument('-c','--convert_binary', required=False, default=False, action='store_true', \
                      help='Convert binary word embedding file into plain text')
  parser.add_argument('-s','--start_silver_learning', required=False, default=False, action='store_true', \
                      help='transfer learning is one')
  parser.add_argument('-t','--datasplit', required=False, default=False, action='store_true', \
                      help='split data into training, testing, and validation')
  parser.add_argument('-j','--path_to_silver_data', required=False, help='Path to silver data')

  oj_args = parser.parse_args(sys.argv[1:])

  print('[INFO]: NTLK Version: %s' % nltk.__version__)
  print('[INFO]: gold standard database...')
  oj_ncbi_disease = NcbiDiseaseCorpus()

  if oj_args.convert_binary and os.path.exists(oj_args.path_to_binary):
    print("[INFO]: Converting binary word embedding into word... ")
    oj_ncbi_disease.convert_bin_to_glove(oj_args.path_to_binary) 

  # unit test
  if oj_args.unit_test:
    oj_ncbi_disease.parse_ncbi_disease_corpus("./unittest/data/NCBI_corpus.txt")
    oj_ncbi_disease.get_statistics(dump_pubmed_id=True, dump_disease_name=True, tag="gold")
    oj_ncbi_disease.convert_in_bert_format()

    # split gold standard data into train/test/valid in the ratio of 6:2:2
    oj_model = Model(0.8, 0.1, 0.1)
    oj_model.create_validation_data(oj_ncbi_disease.get_pubmed_ids())

    # silver standard data
    oj_ncbi_disease.tokenize_gold_disease_name()
    oj_ncbi_disease.parseXMLPubTator("./unittest/data/CDR_TrainingSet.BioC.xml", False)
    oj_ncbi_disease.find_DE_in_corpus()
    oj_ncbi_disease.write_valid_silver_corpus_annotation()
    sys.exit(0)
  
  oj_ncbi_disease.parse_ncbi_disease_corpus(oj_args.path_to_input)
  oj_ncbi_disease.get_statistics(dump_pubmed_id=True, dump_disease_name=True, tag="gold")
  oj_ncbi_disease.convert_in_bert_format()
  
  if oj_args.datasplit:
    oj_model = Model(0.8, 0.1, 0.1)
    oj_model.create_validation_data(oj_ncbi_disease.get_pubmed_ids())
    oj_model.split_validation_data() 

  if oj_args.start_silver_learning:
    print('[INFO]: silver standard database...')
    # Labelled disease name from gold standard data 
    oj_ncbi_disease.tokenize_gold_disease_name()

    if os.path.exists(oj_args.path_to_silver_data):
      print("[INFO]: parsing {}".format(oj_args.path_to_silver_data))
      oj_ncbi_disease.parseXMLPubTator(oj_args.path_to_silver_data, False)
      oj_ncbi_disease.find_DE_in_corpus()
      oj_ncbi_disease.write_valid_silver_corpus_annotation()

    # statistics
    oj_ncbi_disease.compare_pubmed_ids()
    oj_ncbi_disease.compare_diseases_annotation()
    oj_ncbi_disease.get_statistics(dump_pubmed_id=True, dump_disease_name=True, tag="silver")

