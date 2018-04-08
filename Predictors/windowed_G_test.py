from data.import_data import import_data



if __name__ == '__main__':
    tr, te = import_data()

    for s in tr.text[:10]:
        print(word_tokenize(s))
