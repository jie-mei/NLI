import data

for i, (s1, s2, label, feat1, feat2) in enumerate(data.SNLI.parse('validation')):
    print(label)
    print(s1)
    print(s2)
    print(feat1)
    print(feat2)
    print()
    if i >= 3:
        break
