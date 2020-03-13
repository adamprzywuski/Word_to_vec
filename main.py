from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

def teaching():
    f = open("book1.txt", "r")
    f1=open("book2.txt","r")
    contents = f.read()
    contents1=f1.read()

    s = 'Although Shakespeare was married to a woman and fathered three children, Susanna, Hamnet and Judith, people have debated his sexuality.'
    s1 = 'Some people, such as Peter Holland of the Shakespeare Institute at Birmingham University, have argued that Shakespeare was possibly bisexual because of some of the sonnets he wrote that were directed towards young men.'
    s2 = 'About 150 years after Shakespeare died, some writers began to say that the work called "Shakespeare" were not really written by William Shakespeare. They had various reasons for saying this. For example, the person who wrote "Shakespeare" knew a lot about other countries (especially Italy and France), but William Shakespeare never left England. Several other writers of "Shakespeare" have been suggested, such as Francis Bacon, Christopher Marlowe, and Edward de Vere, 17th Earl of Oxford. Most scholars believe that William Shakespeare did write the works that bear his name.'
    t = nltk.word_tokenize(s)
    t1 = nltk.word_tokenize(s1)
    t2 = nltk.word_tokenize(s2)
    t3 = nltk.word_tokenize(contents)
    t4=nltk.word_tokenize(contents1)
    common_texts = [["his", "plays", "are", "of", "different", "kinds", "or", "genres"],
                    ["There", "are", "histories", "tragedies", "and", "comedies"],
                    ["These", "plays", "are", "among", "the", "best", "known", "in", "English", "literature", "and"],
                    ['Good', 'muffins', 'cost', '$', '3', '.', '88', 'in', 'New', 'York', '.',
                     'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.'],
                    ]
    common_texts.append(t)
    common_texts.append(t1)
    common_texts.append(t2)
    common_texts.append(t3)
#    common_texts.append(t4)
    model = Word2Vec(common_texts, size=50, window=5, min_count=1, workers=4)
    model.save("model.model")
    return model


def tests(model):
    print(model.train([["buy", "world"]], total_examples=1, epochs=1))
    vector = model.wv["cost"]
    vector1 = model.wv["years"]
    w1 = "comedies"
    w2 = "tragedies"
    w3 = "genres"
    w4 = "kinds"
    print("Word similar to women: ")
    print(model.wv.most_similar(positive="women"))
    print(model.wv.similarity(w2, w1))
    print(model.wv.similarity(w3, w4))
    print(model.wv.similarity(w1, w4))
    print(model.wv.similarity("you", "building"))
    print(model.wv.similarity("me", "I"))
    print(vector)
    print("----------")
    print(vector1)

if __name__ == '__main__':
    #model=teaching()
    model=Word2Vec.load("model.model")
    print("Teaching/Loading has completed")
    tests(model)