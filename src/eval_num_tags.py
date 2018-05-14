import data

train_data = data.load_dataset('SNLI', 'train', 'GloVeNorm', 6523)
val_data   = data.load_dataset('SNLI', 'validation', 'GloVeNorm', 6523)
test_data  = data.load_dataset('SNLI', 'test', 'GloVeNorm', 6523)

tags = set()  # type: ignore

for dset in [train_data, val_data, test_data]:
    for feat in dset.x1_feats + dset.x2_feats:
        for tag in feat[1]:
            tags.add(tag)

print(tags)
print(len(tags))
