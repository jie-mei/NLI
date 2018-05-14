import data

train_data = data.load_dataset('SNLI', 'train', 'GloVeNorm', 6523)
val_data   = data.load_dataset('SNLI', 'validation', 'GloVeNorm', 6523)
test_data  = data.load_dataset('SNLI', 'test', 'GloVeNorm', 6523)

print(train_data.x1_words[0])
print(train_data.x1_ids[0])
print(train_data.x1_feats[0])

if False:
    tags = set()  # type: ignore

    for dset in [train_data, val_data, test_data]:
        for id_, feat in zip(dset.x1_ids + dset.x2_ids, dset.x1_feats + dset.x2_feats):
            if len(id_) + len(feat[0]) != len(feat[1]):
                raise ValueError
            for temp in feat[0]:
                if max(temp) >= len(feat[1]):
                    raise ValueError

