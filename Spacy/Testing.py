import numpy as np
import spacy
from spacy import displacy
from tqdm import tqdm

from Spacy.my_sentencizer import *

en = spacy.load("en_core_web_sm")
nlp = spacy.load("../Spacy/Training/Output/TaggerMorpherParser98/model-best")
nlp.add_pipe("hsb_split", before="tagger")
displacy.serve(nlp("Kocor je z njewšědnej pilnosću dźěłał, wozbožujo swój lud z wulkej syłu małych a wjetšich składbow. "), port=3000, style="dep")

spacy.require_gpu()


nlp = spacy.load("../Spacy/Training/Output/TaggerMorpherParser98/model-best")
nlp.add_pipe("hsb_split", before="tagger")

from spacy_conllu import ConlluParser, init_parser

parser = ConlluParser(init_parser("hsb"))
clean, corpus = parser.parse_conll_file_as_spacy("../Data/hsb_UD.conllu", input_encoding="utf-8", pair=True, combine=False)

results = {"FullUD96":0, "NewTagger94":0, "Invtagger90":0, "TaggerMorpherParser98":0}

models = {model:spacy.load(f"../Spacy/Training/Output/{model}/model-best") for model in results.keys()}

# for truth, doc in tqdm(zip(corpus, clean)):
#
#     for m in results.keys():
#         res = models[m](doc)
#         check = ["tag_"]
#         for t1, t2 in zip(truth, res):
#             a = [t1.tag_ == t2.tag_ for pr in check]
#             add = np.count_nonzero(np.where(a, 1, 0))/len(check)
#             # print(a)
#             results[m]+=add/len(res)
#
# results = {k: v/len(clean) for k, v in results.items()}
# print(results)
