from extract_text import train_resumes, test_resumes

train_resumes_lower = []
for resume in train_resumes:
    train_resumes_lower.append(resume.lower())


test_resumes_lower = []
for resume in test_resumes:
    test_resumes_lower.append(resume.lower())

print(train_resumes_lower[0])
print('===========')
print(test_resumes_lower[0])
