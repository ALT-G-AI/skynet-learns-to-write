import argparse
import os

import pandas as pd
from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

NAN_WORD = "_NAN_"


# This is based on code by Pavel Ostyakov (https://github.com/PavelOstyakov/toxic/tree/master/tools)
# which is distributed with an MIT License

def translate(sentence, language):
    if hasattr(sentence, "decode"):
        sentence = sentence.decode("utf-8")

    text = TextBlob(sentence)
    try:
        text = text.translate(to=language)
        text = text.translate(to="en")
    except NotTranslated:
        pass

    return str(text)


def main():
    parser = argparse.ArgumentParser("Script for extending training dataset")
    parser.add_argument("train_file_path")
    parser.add_argument("--languages", nargs="+", default=["es", "de", "fr"])
    parser.add_argument("--thread-count", type=int, default=300)
    parser.add_argument("--result-path", default="extended_data")

    args = parser.parse_args()

    train_data = pd.read_csv(args.train_file_path)
    sentence_list = train_data["text"].fillna(NAN_WORD).values

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    parallel = Parallel(args.thread_count, backend="threading", verbose=5)
    for language in args.languages:
        print('Translate sentence using "{0}" language'.format(language))
        translated_data = parallel(delayed(translate)(sentence, language) for sentence in sentence_list)
        train_data["text"] = translated_data

        result_path = os.path.join(args.result_path, "train_" + language + ".csv")
        train_data.to_csv(result_path, index=False)


if __name__ == "__main__":
    main()
