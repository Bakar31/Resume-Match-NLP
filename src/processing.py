import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

needless_words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
'V', 'W', 'X', 'Y', 'Z']
punctuations = list('''!()-[]{};:'"\,<>./?@#$%^&*_~''')

def text_processing(resume):   

    resume = nlp(resume)
    token_list = []
    for token in resume:
        token_list.append(token.text)

    filtered_sentence =[] 
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 

    # further filter
    filtered_sentence_2 = []
    for word in filtered_sentence:
        if word not in needless_words:
            filtered_sentence_2.append(word)

    filtered_sentence_3 = []
    for word in filtered_sentence_2:
        if word not in punctuations:
            filtered_sentence_3.append(word)
    
    Stem_words = []
    sentence = ' '.join(filtered_sentence_3)
    doc = nlp(sentence)
    for word in doc:
        Stem_words.append(word.lemma_)

    main_text = ' '.join(Stem_words)
    return main_text

