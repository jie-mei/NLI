import data

train_data = data.load_dataset('SNLI', 'train', 'GloVeNorm', 6523)
valid_data = data.load_dataset('SNLI', 'validation', 'GloVeNorm', 6523)
test_data = data.load_dataset('SNLI', 'test', 'GloVeNorm', 6523)

tags = set()
for d in [train_data, valid_data, test_data]:
    for t_list in d.x1_feats + d.x2_feats:
        for t in t_list:
            tags.add(t)

print(tags)
print(len(tags))

