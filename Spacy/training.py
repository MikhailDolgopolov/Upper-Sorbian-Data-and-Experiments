import random

import pandas as pd
import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import minibatch
from thinc.schedules import compounding

raw_data = pd.read_csv("../Data/UPOS_UD-data.csv")
# print(raw_data)
size = len(raw_data)
# from spacy_conll import init_parser
# from spacy_conll.parser import ConllParser
#
#
# nlp = ConllParser(init_parser("hsb", "spacy"))
#
# doc = nlp.parse_conll_file_as_spacy("../Data/hsb_UD.conllu")
train_data = []
upos = raw_data["upos"].unique()
for i in range(5):
    form = raw_data.at[i, "word"]
    spec = {"pos": raw_data.at[i, "upos"]}
    train_data.append((form, spec))
print(train_data)

def main(output_dir=None, n_iter=25):
    nlp = spacy.blank("hsb")
    nlp.add_pipe('tagger')
    tagger = nlp.get_pipe("tagger")
    examples = []
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        examples.append(example)

    for pos in upos:
        tagger.add_label(pos)
    optimizer = nlp.initialize()
    for i in range(n_iter):
        random.shuffle(examples)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, losses=losses)
        print("Losses", losses)

main()