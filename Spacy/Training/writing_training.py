from spacy.tokens import DocBin

from spacy_conllu import ConlluParser, init_parser

parser = ConlluParser(init_parser("hsb"))
clean, corpus = parser.parse_conll_file_as_spacy("../../Data/hsb_UD.conllu", input_encoding="utf-8", pair=True, combine=False)


train_bin = DocBin(docs=clean)
train_bin.to_disk("./data.spacy")
corpus_bin = DocBin(docs=corpus)
corpus_bin.to_disk("./corpus.spacy")