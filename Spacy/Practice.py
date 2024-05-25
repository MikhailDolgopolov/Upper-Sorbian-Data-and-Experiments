from pathlib import Path

import numpy as np
import spacy
from spacy import displacy
from spacy.scorer import Scorer
from tqdm import tqdm

from Spacy.my_sentencizer import *

en = spacy.load("en_core_web_sm")
nlp = spacy.load("../Spacy/Training/Output/TaggerMorpherParser98/model-best")
nlp.add_pipe("hsb_split", before="tagger")
doc = nlp("Kocor je z njewšědnej pilnosću dźěłał, wozbožujo swój lud z wulkej syłu małych a wjetšich składbow. ")
svg = displacy.render(doc,style="dep")
file_name = '-'.join([w.text for w in doc[:min(len(doc), 6)] if not w.is_punct]) + ".svg"
output_path = Path("../Images/" + file_name)
output_path.open("w", encoding="utf-8").write(svg)


# nlp = spacy.load("../Spacy/Training/Output/TaggerMorpherParser98/model-best")
# nlp.add_pipe("hsb_split", before="tagger")
#
# from spacy_conllu import ConlluParser, init_parser
#
# parser = ConlluParser(init_parser("hsb"))
# clean, corpus = parser.parse_conll_file_as_spacy("../Data/hsb_UD.conllu", input_encoding="utf-8", pair=True, combine=False)
#
# results = {"FullUD96":0, "NewTagger94":0, "Invtagger90":0, "TaggerMorpherParser98":0}
#
# models = {model:spacy.load(f"../Spacy/Training/Output/{model}/model-best") for model in results.keys()}
#
# scorer = Scorer(nlp)
