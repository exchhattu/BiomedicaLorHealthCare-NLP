#!/usr/bin/python3

'''
Written: Rojan Shrestha PhD
Mon Sep 16 15:04:45 2019
'''

import os, sys, shutil
import random

class Model:

  def __init__(self, fo_train, fo_valid, fo_test):
    self._fo_train  = fo_train 
    self._fo_valid  = fo_valid
    self._fo_test   = fo_test

    # indexes 
    self._ts_train = []
    self._ts_valid = [] 
    self._ts_test  = [] 

  def create_validation_data(self, ts_rows):
    in_seed = 89
    ts_row_idxes = [i for i in range(0, ts_rows.shape[0])] 
    random.Random(in_seed).shuffle(ts_row_idxes) 

    in_train    = int(self._fo_train * len(ts_row_idxes))
    in_valid    = int(self._fo_valid * len(ts_row_idxes))
    in_test     = int(self._fo_test * len(ts_row_idxes))

    # in the case if float and int conversion lose some data
    in_error    = in_train + in_valid + in_test
    in_diff     = len(ts_row_idxes) - in_error
    in_train    = in_train + in_diff

    print("[Updates]: {0:0.2f}% train => {1:d}".format(self._fo_train, in_train))
    print("[Updates]: {0:0.2f}% valid => {1:d}".format(self._fo_valid, in_valid))
    print("[Updates]: {0:0.2f}% test  => {1:d}".format(self._fo_test, in_test))

    in_tr_idxes = ts_row_idxes[:in_train] 
    in_va_idxes = ts_row_idxes[in_train:in_train+in_valid] 
    in_te_idxes = ts_row_idxes[-in_test:] 

    self._ts_train = ts_rows[in_tr_idxes]
    self._ts_valid = ts_rows[in_va_idxes]
    self._ts_test =  ts_rows[in_te_idxes]

  def split_validation_data(self, root_path=os.getcwd()): 
    ts_dirs = ["train", "valid", "test"]
    for st_dname in ts_dirs:
      self.copy_data(root_path=root_path, data_type=st_dname) 

  def copy_data(self, root_path=os.getcwd(), data_type="train"): 
    try:
      path = os.path.join(root_path, data_type)
      if os.path.exists(path): shutil.rmtree(path)
      os.makedirs(path)
   
      ts_data = []
      if data_type=="train":
        ts_data = self._ts_train
      elif data_type=="valid":
        ts_data = self._ts_valid
      elif data_type=="test":
        ts_data = self._ts_test 

      # Copy files in respective directory 
      ts_formats = ["txt", "ana"]
      for st_format in ts_formats:
        for st_dir in ts_data: 
          f_spath = os.path.join(root_path, "%s.%s" %(st_dir, st_format))
          f_dpath = os.path.join(root_path, data_type)
          if os.path.isfile(f_spath): 
            shutil.copy(f_spath, f_dpath)
            os.remove(f_spath)
    except:
      print("[FATAL] creating data for validation is unsuccessful.")
      sys.exit(0)

