from sklearn.preprocessing import OneHotEncoder

def clean_word(w):
    return ''.join([ c.lower() for c in w if c.isalpha() ])

class Data(object):
    # sentence_length=50 is big enough for the vast majority of the data
    def __init__(self, processed_file = 'data_module/a.txt', sentence_length=50):
        data = list()
        with open(processed_file, 'r', encoding='utf8') as inf:
            # clean up all of the words - removing nonalphabetic characters
            for line in inf:
                clean_line = [clean_word(w) for w in line.split(' ')]
                clean_line.append('.') # give every sentence a termination symbol

                # sentences need to have a fixed length for most of sklearn's stuff
                if len(clean_line) > sentence_length:
                    continue
                elif len(clean_line) < sentence_length:
                    clean_line.extend('\0'*(sentence_length - len(clean_line)))

                assert len(clean_line) == sentence_length
                data.append(clean_line)

        # make dictionary of unique words, each with a number to represent them
        self.words = {'\0': 0}
        index = 1
        for line in data:
            for word in line:
                if (word in self.words):
                    continue
                else:
                    self.words[word] = index
                    index += 1

        # replace with numbers
        proc_data = list()
        for line in data:
            proc_data.append([self.words[w] for w in line])

        # one hot encoding
        self.enc = OneHotEncoder(dtype=int)
        self.all_data = self.enc.fit_transform(proc_data)


if __name__ == '__main__':
    d = Data()
    print(d.all_data[0])
