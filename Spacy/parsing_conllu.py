import random
from pathlib import Path

import pandas as pd
import numpy as np
import spacy
import time

from spacy.pipeline.morphologizer import Scorer
from tqdm import tqdm

from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.util import minibatch
from thinc.schedules import compounding

from spacy_conllu import init_parser
from spacy_conllu.parser import ConlluParser

spacy.require_gpu()

start_time = time.time()
parser = ConlluParser(init_parser("hsb"))

doc, training_doc = parser.parse_conll_file_as_spacy("../Data/hsb_UD.conllu", input_encoding="utf-8", pair=True)
print(type(doc), type(training_doc))
print(f"read time: {time.time() - start_time}")

# upos = pd.read_csv("../Data/UPOS_UD-data.csv")["upos"].unique()
#
# bin = DocBin(attrs=["SENT_START", "DEP"])
# bin.add(doc)
# copy_doc = next(bin.get_docs(doc.vocab))
# l=12
# for token in copy_doc[:12]:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)
# copy_doc = Doc.from_array(doc, ["token"],doc.to_array(py_attr_ids=["token"]))

all_examples = Example(training_doc, doc)
train_data = all_examples.split_sents()



def train(pipes, output_dir=None, n_iter=25, early_break=True):
    from statistics import harmonic_mean
    nlp = spacy.blank("hsb")
    for pipe in pipes:
        nlp.create_pipe(pipe)
        nlp.add_pipe(pipe)
    scorer_dict={ "trainable_lemmatizer":"lemma_acc", "morphologizer":"morph_acc", "tagger":"tag_acc"}
    optimizer = nlp.initialize(lambda: random.sample(train_data, int(len(train_data)/4)))
    scorer = Scorer(nlp)
    scores=[]
    max_accuracy=0
    rampup=int(np.sqrt(n_iter))
    pipe = pipes[0] if len(pipes)==1 else ", ".join([p[:4] for p in pipes])
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        # batch up the examples using spaCy's minibatch compounding(4.0, 32.0, 1.001)

        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.01))
        for batch in batches:
            # exs = []
            # for j in range(si):
            #     # print(batch[j])
            #     exs.append(Example.from_dict(nlp.make_doc(batch[j][0]), batch[j][1]))
            losses = nlp.update(batch, losses=losses, sgd=optimizer, drop=0.1)

        check = [Example(nlp(point.predicted), point.reference) for point in random.sample(train_data, int(len(train_data)/4))]
        score_result = scorer.score(check)
        a = np.array([score_result[scorer_dict[part]] for part in pipes])
        res=harmonic_mean(a)
        max_accuracy = max([*scores,0])
        print(f"Losses {i}: {[losses[p] for p in pipes]}, {pipe} accuracy: {res}")
        if len(scores)>rampup: scores.pop(0)
        scores.append(res)
        if np.mean(scores)<max_accuracy-0.01 and i>rampup and early_break:
            print(f"Saving {scores[-1]} accuracy")
            break

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


output_dir="Models/MyBigTagger"
train(["tagger"], output_dir, 25, early_break=False)
# output_dir="Models/MyComb_TagMorph"
# train(["tagger", "morphologizer"], output_dir, 100)
# output_dir="Models/MyComb_TagLemmMorph"
# train(["tagger", "trainable_lemmatizer", "morphologizer"], output_dir, 100)
