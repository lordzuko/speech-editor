import re
import pandas as pd

df = pd.read_csv("/work/tc046/tc046/lordzuko/data/blizzard_2013/BC2013_seg_v1/testset-transcript.csv", delimiter="|")

df.columns = ["wav_name", "abc", "text", "random"]
df = df[["wav_name", "text"]]
df["utt_len"] = df["text"].apply(lambda x: len(x.split(" ")))

df.to_csv("/work/tc046/tc046/lordzuko/work/speech-editor/data/text.csv", index=False)
