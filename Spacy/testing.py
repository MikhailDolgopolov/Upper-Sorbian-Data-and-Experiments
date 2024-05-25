import time
from functools import reduce
from pprint import pprint

import spacy
import pandas as pd

from Spacy.utils import get_hsb_model

pd.set_option('display.max_columns', None)
from pandas import Series
from spacy.scorer import Scorer
from spacy.training import Example

from Spacy.my_sentencizer import *
from spacy_conllu import init_parser, ConlluParser

file_name = "../Data/citanka-prose.txt"

start_time = time.time()
nlp = spacy.load("../Spacy/Training/Output/TaggerMorpherParser98/model-best")
nlp.add_pipe("hsb_split", before="tagger")

parser = ConlluParser(init_parser("hsb"))

clean_doc, truth_doc = parser.parse_conll_file_as_spacy("../Data/hsb_UD.conllu", input_encoding="utf-8", pair=True,
                                                        combine=True)

# pprint(clean_doc.to_json())

m_names = ["NewTry", "ProseTrifecta"]

models = {model:get_hsb_model(model) for model in m_names}
examples = {m: Example(models[m](clean_doc), truth_doc).split_sents() for m in m_names}

scorers = {m: Scorer(models[m]) for m in m_names}


result = {m: {k: v for k, v in scorers[m].score(examples[m]).items() if v not in [1, 0] and isinstance(v, float)} for m
          in m_names}

c_c = [[c for c in result[m].keys()] for m, r in result.items()]

# intersection
columns = list(reduce(set.intersection, [set(item) for item in c_c]))

df_dict = {c: [result[m][c] for m in m_names] for c in columns}

df = pd.DataFrame.from_dict(df_dict)
df.index = Series(m_names)

print(df)
