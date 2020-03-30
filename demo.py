from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

def teaching():
    f1 = open("book1.txt", "r")
    f2=open("book2.txt","r")
    f3=open("book3.txt","r")
    f4=open("book3.txt","r")

    contents4 = f4.read()
    contents1 = f1.read()
    contents2=f2.read()
    contents3 = f3.read()

    t1 = nltk.word_tokenize(contents1)
    t2=nltk.word_tokenize(contents2)
    t3 = nltk.word_tokenize(contents3)
    t4 = nltk.word_tokenize(contents4)

    common_texts.append(t1)
    common_texts.append(t2)
    common_texts.append(t3)
    common_texts.append(t4)

    modelCBOW = Word2Vec(common_texts, size=50, window=5, min_count=1, workers=4)
    modelSkipGram= Word2Vec(common_texts, size=50, window=5, min_count=1, workers=4,sg=2)
    modelCBOW.save("CBOW.model")
    modelSkipGram.save("SkipGram.model")
    return modelCBOW,modelSkipGram


def tests(modelCBOW,modelSkipGram):
    while(1):

        print("Please select what you want to check:")
        switch=input("1.How similar is this 2 words to each others \n 2.The most familiar words:\n 3.Exit")
        if(switch=='1'):
            first=input("First_word:")
            second=input("Second word:")
            print(modelCBOW.wv.similarity(first, second))

            print(modelSkipGram.wv.similarity(first, second))
        elif(switch=='2'):
            first=input("Word:")
            print(modelCBOW.wv.most_similar(positive=first))
            print(modelSkipGram.wv.most_similar(positive=first))
        elif(switch=='3'):
            exit()
        else:
            print("Try again")





if __name__ == '__main__':
    print("---DEMO---")
    modelCBOW,modelSkipGram=teaching()
    #modelCBOW=Word2Vec.load("CBOW.model")
    #modelSkipGram=Word2Vec.load("SkipGram.model")
    print("Teaching/Loading has completed")
    #tests(modelCBOW,modelSkipGram)