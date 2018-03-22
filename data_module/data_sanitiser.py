import string


def sanitise(infile, outfile):
    """
    Choose lines which have at least 1 full stop, and contain no special
    characters other than `,'"`.
    :param infile: input file to read from
    :param outfile: output to write to.
    """
    allowed = string.ascii_letters + string.whitespace + ',\'\"'
    print('allowed', allowed)
    out_set = set()
    with open(infile, 'r', encoding="utf8") as inf:
            for line in inf:
                sentences = line.split('.')
                if len(sentences) == 1:
                    continue
                for sent in sentences:
                    if all([c in allowed for c in sent]):
                        words = sent.split(' ')
                        if len(words) >= 5:
                            out_set.add(sent.strip()+'\n')
    with open(outfile, 'w') as ouf:
        ouf.writelines(out_set)

if __name__ == '__main__':
    sanitise('plaintext-olddata.txt', 'a.txt')
