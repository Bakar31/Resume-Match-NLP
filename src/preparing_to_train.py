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

'''print(len(processed_resumes_train))
print(len(processed_resumes_test))'''
#print(processed_resumes_train[1])
#print(processed_resumes_test[0])

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

def dataframe(resume_list, df):
    resumes =  pd.DataFrame(resume_list, columns = ['resumes'])
    dataframe = pd.concat([df, resumes], axis = 1)
    dataframe.drop('CandidateID', axis = 1, inplace = True)
    return dataframe

train_df = dataframe(processed_resumes_train, train)
test_df = dataframe(processed_resumes_test, test)

'''print(train_df.head())
print(test_df.head())'''
