from datasets import load_dataset

train_ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split ='train')


for eg in train_ds:
    print(eg)
    break

