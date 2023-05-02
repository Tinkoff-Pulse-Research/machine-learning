import json
import re
import typing

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Preprocessor:
    def __init__(self):
        self._stopwords = set(stopwords.words("russian"))
        self._lemmatizer = WordNetLemmatizer()
        self._stemmer = PorterStemmer()

    def _lemmatize(self, text: str) -> str:
        """Lemmatizes text"""
        return " ".join(
            [
                self._lemmatizer.lemmatize(self._stemmer.stem(w))
                for w in word_tokenize(text)
            ]
        )

    @staticmethod
    def _keep_only_letters(text: str) -> str:
        """Removes everything except of latin and cyrillic letters"""
        return re.sub(r"[^a-zA-Zа-яА-Я ]", "", text)

    @staticmethod
    def _remove_roman(text: str) -> str:
        """Removes roman digits"""
        return re.sub(r"\b[xvi]+\b", "", text)

    @staticmethod
    def _normalize_whitespaces(text: str) -> str:
        """Replaces multiple whitespaces with single"""
        return re.sub(r" {2,}", " ", text)

    @staticmethod
    def _remove_tickers(text: str) -> str:
        """Removes tickers like $AMZN"""
        return re.sub(r"\{\$.*?\}", "", text)

    @staticmethod
    def _remove_mentions(text: str) -> str:
        """Removes mentions"""
        return re.sub(r"\@[0-9a-f-]+", "", text)

    @staticmethod
    def _remove_links(text: str) -> str:
        """Removes links"""
        return re.sub(
            r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            "",
            re.sub(r"\[&[0-9a-f-]+?\]\(https:.+?\)", "", text),
        )

    def _remove_stopwords(self, text: str) -> str:
        """Removes common stopwords"""
        return " ".join(
            filter(
                lambda token: token.strip() not in self._stopwords and token.strip(),
                text.split(),
            )
        )

    def _preprocess(self, text: str) -> str:
        """Combines all methods of preprocessing"""
        return self._lemmatize(
            self._remove_stopwords(
                self._normalize_whitespaces(
                    self._remove_roman(
                        self._keep_only_letters(
                            self._remove_links(
                                self._remove_mentions(
                                    self._remove_tickers(text.lower().replace("ё", "е"))
                                )
                            )
                        )
                    )
                )
            )
        )

    def preprocess(self, texts: typing.List[str]) -> typing.List[str]:
        return list(map(self._preprocess, texts))


if __name__ == "__main__":
    preprocessor = Preprocessor()

    import csv

    with open("tinkoff_kochubey_pulse_slang.csv", "r") as f:
        reader = csv.reader(f)
        slang = list(reader)

    slang = [i[0] for i in slang][1:]

    with open("res.json", "w") as f:
        json.dump(preprocessor.preprocess(slang), f, ensure_ascii=False, indent=4)
