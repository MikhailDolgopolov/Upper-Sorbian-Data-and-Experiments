from spacy.tokens import DocBin

from spacy_conllu import ConlluParser, init_parser

parser = ConlluParser(init_parser("hsb"))
clean, corpus = parser.parse_conll_file_as_spacy("../../Data/hsb_UD.conllu", input_encoding="utf-8", pair=True, combine=False)


corpus_bin = DocBin(docs=corpus)
corpus_bin.to_disk("corpora/UD.spacy")