# required libraries
import os
import pandas as pd
import numpy as np
from pdfminer import high_level

#paths
train_path = "dataset/trainResumes/"
test_path = "dataset/testResumes/"

# epty list for resumes text
train_resumes = []
test_resumes = []

# pdf2string
def pdf2string(path, resumes):
    for i in os.listdir(path):
        main_path = path+i
        text = high_level.extract_text(main_path)
        str_list = text.split()
        str_list = str_list[25:]
        string = ' '.join(str_list)
        resumes.append(string)

pdf2string(train_path, train_resumes)
pdf2string(test_path, test_resumes)

print(train_resumes[0])
print('==================')
print(test_resumes[0])