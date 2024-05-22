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
# from spacy_conllu import init_parser
# from spacy_conllu.parser import ConlluParser
#
#
# nlp = ConlluParser(init_parser("hsb", "spacy"))
#
# doc = nlp.parse_conll_file_as_spacy("../Data/hsb_UD.conllu")
train_data = []
upos = raw_data["upos"].unique()
for i in range(0,16,4):
    forms = [raw_data.at[j, "word"] for j in range(i,i+4)]
    specs = {"pos": [raw_data.at[j, "upos"] for j in range(i, i+4)]}
    train_data.append((" ".join(forms), specs))

# print(train_data)

def main(output_dir=None, n_iter=25):
    nlp = spacy.blank("hsb")
    nlp.add_pipe('tagger')
    # tagger = nlp.get_pipe("tagger")
    examples = []
    for text, annotations in train_data:
        # print(nlp.make_doc(text), annotations)
        example = Example.from_dict(nlp.make_doc(text), annotations)
        examples.append(example)

    for pos in upos:
        nlp.get_pipe("tagger").add_label(pos)
    optimizer = nlp.initialize()
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        # batch up the examples using spaCy's minibatch compounding(4.0, 32.0, 1.001)
        si = 2
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            print(type(batch), type(batch[0]), type[batch[0][0]])
            # print(batch[1])
            # exs = []
            # for j in range(si):
            #     # print(batch[j])
            #     exs.append(Example.from_dict(nlp.make_doc(batch[j][0]), batch[j][1]))
            losses = nlp.update([*batch], losses=losses, sgd=optimizer)
        print("Losses", losses)

main("/Data/Second/")