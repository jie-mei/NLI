import data

with open('s1.txt', 'w') as f:
    for i, (s1, s2, label, feat1, feat2) in enumerate(data.SNLI.parse('train')):
        f.write(s1 + '\n')
