import spacy
import torch
from spacy.pipeline.tagger import Scorer
from spacy.tokens import DocBin
from spacy.training import Example


output_dir="Models/MyTaggerTest"

nlp = spacy.load(output_dir)
scorer = Scorer(nlp)

from spacy_conllu import init_parser
from spacy_conllu.parser import ConlluParser

spacy.require_gpu()

# parser = ConlluParser(init_parser("hsb"))
#
# doc = parser.parse_conll_file_as_spacy("../Data/hsb_UD.conllu", input_encoding="utf-8")
#
# all_examples = Example(nlp(doc), doc)
# val_data = all_examples.split_sents()
#
# print(scorer.score(val_data)["tag_acc"])

doc = nlp("A skutkujće, zo njejsće na swojej zemi nic jeničcy doma, ale tež z knjezom!")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)