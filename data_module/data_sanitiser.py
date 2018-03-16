import string


def sanitise(infile, outfile):
	allowed = string.ascii_letters + string.whitespace + ',\'\"'
	print('allowed', allowed)
	with open(infile, 'r', encoding="utf8") as inf:
		with open(outfile, 'w') as ouf:
			for line in inf:
				splitted = line.split('.')
				if len(splitted) == 1:
					continue
				for sent in splitted:
					if all([c in allowed for c in sent]):
						words = sent.split(' ')
						if len(words) >= 5:
							ouf.write(sent.strip() + '\n')


if __name__ == '__main__':
	sanitise('plaintext-olddata.txt', 'a.txt')
