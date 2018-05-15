import data

train_data = data.load_dataset('SNLI', 'train', 'GloVeNorm', 6523)
val_data   = data.load_dataset('SNLI', 'validation', 'GloVeNorm', 6523)
test_data  = data.load_dataset('SNLI', 'test', 'GloVeNorm', 6523)


# Check if all template has the same size
def check_temp_size(temp_size=4, print_first_n=3):
    i = 0
    for dset in [train_data, val_data, test_data]:
        for feat in dset.x1_feats + dset.x2_feats:
            if print_first_n > i:
                print(feat)
            for temp in feat[0]:
                if len(temp) != temp_size:
                    raise ValueError
            i += 1
check_temp_size()

"""
for dset in [train_data, val_data, test_data]:
    for id_, feat in zip(dset.x1_ids + dset.x2_ids, dset.x1_feats + dset.x2_feats):
        if len(id_) + len(feat[0]) != len(feat[1]):
            raise ValueError
        for temp in feat[0]:
            if max(temp) >= len(feat[1]):
                raise ValueError
"""

