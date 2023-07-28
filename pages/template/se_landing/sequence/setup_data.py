import re
import pandas as pd

df = pd.read_csv("/home/lordzuko/work/speech-editor/notebooks/trainset-transcript.csv", delimiter="|")

df.columns = ["wav_name", "abc", "text", "random"]
df = df[["wav_name", "text"]]
df["utt_len"] = df["text"].apply(lambda x: len(x.split(" ")))

df.to_csv("/home/lordzuko/work/speech-editor/notebooks/train.csv", index=False)
