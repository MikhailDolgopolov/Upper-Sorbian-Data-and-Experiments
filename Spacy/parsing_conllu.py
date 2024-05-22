import random
from pathlib import Path

import pandas as pd
import spacy
import time

from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.util import minibatch
from thinc.schedules import compounding

from spacy_conllu import init_parser
from spacy_conllu.parser import ConlluParser

spacy.require_gpu()

start_time = time.time()
nlp = ConlluParser(init_parser("hsb"))

doc = nlp.parse_conll_file_as_spacy("../Data/hsb_UD.conllu", input_encoding="utf-8")
# doc = spacy.blank("en")("I like python")
print(f"read time: {time.time() - start_time}")

upos = pd.read_csv("../Data/UPOS_UD-data.csv")["upos"].unique()

bin = DocBin(attrs=["SENT_START", "DEP"])
bin.add(doc)
copy_doc = next(bin.get_docs(doc.vocab))
print(len(copy_doc), len(doc))
# copy_doc = Doc.from_array(doc, ["token"],doc.to_array(py_attr_ids=["token"]))

all_examples = Example(copy_doc, doc)
train_data = all_examples.split_sents()
print(type(train_data[0]))

def main(pipe, output_dir=None, n_iter=25):
    nlp = spacy.blank("hsb")
    tagger = nlp.create_pipe(pipe)
    nlp.add_pipe(pipe)
    optimizer = nlp.initialize(lambda: random.sample(train_data, int(len(train_data)/5)))
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        # batch up the examples using spaCy's minibatch compounding(4.0, 32.0, 1.001)
        si = 6
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            exs = []
            # for j in range(si):
            #     # print(batch[j])
            #     exs.append(Example.from_dict(nlp.make_doc(batch[j][0]), batch[j][1]))
            losses = nlp.update(batch, losses=losses, sgd=optimizer)
        print(f"Losses {i}", losses)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


output_dir="Models/MyMorph2"
main("morphologizer", output_dir, 20)
