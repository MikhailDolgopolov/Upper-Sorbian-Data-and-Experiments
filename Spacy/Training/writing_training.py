from spacy.tokens import DocBin

from Spacy.utils import spacify_text_file, split_doc
from spacy_conllu import ConlluParser, init_parser

parser = ConlluParser(init_parser("hsb"))
corpus = spacify_text_file("../../Data/citanka-prose.txt")
corpus = split_doc(corpus)
for s in corpus[:8]:
    print(s)
