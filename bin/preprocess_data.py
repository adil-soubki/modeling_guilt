#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepares data for finetuning.

Example Usage:
    $ preprocess_data.py data/SuspectGuilt/Data/model_data.tsv -o data/splits/
"""
import argparse
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import transformers as tf


# === TAKEN FROM modeling/model/utils.py === #
string_split = lambda string: [
    (m.group(0), m.start(), m.end() - 1) for m in re.finditer(r'\S+', string)
]

def wordpiece_with_indices(tokenizer, word, start_index=0):
    tokens = tokenizer.tokenize(word)
    output = []
    for tok in tokens:
        if tok[:2] == "##":
            output.append((tok, start_index, start_index + len(tok) - 2))
            start_index += len(tok) - 2
        else:
            output.append((tok, start_index, start_index + len(tok)))
            start_index += len(tok)
    return output
# === TAKEN FROM modeling/model/utils.py === #


def parse_story_tokens(
    string: str,
    tokenizer: tf.PreTrainedTokenizer
) -> list[tuple[str, int, int]]:  # [(token, start, end), ...]
    tokens = string_split(string)
    wordpiece_tokens = sum(
        [wordpiece_with_indices(tokenizer, tok, start) for tok, start, end in tokens], []
    )
    return wordpiece_tokens


def parse_highlight_strs(strings: list[str]) -> list[list[int]]:
    def str_to_int_list(s: str) -> list[int]:
        return [int(c) for c in s]
    return list(map(str_to_int_list, strings))


#  def preprocess_data(df: pd.DataFrame) -> list[dict[str, Any]]:
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = tf.AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased",
        do_lower_case=True
    )
    ret = []
    for sid, sdata in df.groupby("story_id"):
        assert len(sdata.story_clean.unique()) == 1
        story_clean = sdata.story_clean.iloc[0]
        story_tokens = parse_story_tokens(story_clean, tokenizer)
        entry = {
            "story_id": sid,
            "story_clean": story_clean,
        }
        for trial_type, tdata in sdata.groupby("trial_type"):
            assert all(len(bd) == len(story_clean) for bd in tdata.Bin_data)
            mean_slider_val = tdata.slider_val.mean()
            mean_token_highlights = [
                [
                    np.mean(hl[start:end]) for _, start, end in story_tokens
                ] for hl in parse_highlight_strs(tdata.Bin_data)
            ]
            mean_token_highlights = list(np.mean(mean_token_highlights, axis=0))
            assert len(mean_token_highlights) == len(story_tokens)
            #  XXX: old version
            #  mean_highlights = np.array(parse_highlight_strs(tdata.Bin_data)).mean(axis=0)
            if trial_type == "author_belief":
                entry["author_belief"] = mean_slider_val
                entry["author_belief_highlight"] = mean_token_highlights
            elif trial_type == "suspect_committedCrime":
                entry["suspect_committedCrime"] = mean_slider_val
                entry["suspect_committedCrime_highlight"] = mean_token_highlights
            else:
                raise ValueError(f"unknown trial type: {trial_type}")
        ret.append(entry)
    return pd.DataFrame(ret)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("-o", "--outdir", required=True)
    args = parser.parse_args()

    df = preprocess_data(pd.read_csv(args.path, sep="\t"))
    train, dev, test = np.split(
        df.sample(frac=1, random_state=42),
        [int(0.8 * len(df)), int(0.9 * len(df))]
    )
    for split, data in zip(("train", "dev", "test"), (train, dev, test)):
        outpath = os.path.join(args.outdir, f"{split}.jsonl")
        data.to_json(outpath, orient="records", lines=True)
        print(f"wrote: {outpath}")


if __name__ == "__main__":
    main()
