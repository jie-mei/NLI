import data

train_tags = data.load_dataset('SNLI', 'train', 'GloVeNorm').TAGS
val_tags   = data.load_dataset('SNLI', 'validation', 'GloVeNorm').TAGS
test_tags  = data.load_dataset('SNLI', 'test', 'GloVeNorm').TAGS

print(train_tags)
tags = {**train_tags, **val_tags, **test_tags}

print(tags)
print(len(tags))
