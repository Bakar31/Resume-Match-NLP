import pandas as pd
from extract_text import train_resumes, test_resumes
from processing import text_processing

processed_resumes_train = []
processed_resumes_test = []

for  resume in train_resumes:
    processed_resume = text_processing(resume)
    processed_resumes_train.append(processed_resume)

for  resume in test_resumes:
    processed_resume = text_processing(resume)
    processed_resumes_test.append(processed_resume)

print(len(processed_resumes_train))
print(len(processed_resumes_test))
print(processed_resumes_train[0])
print(processed_resumes_test[0])

