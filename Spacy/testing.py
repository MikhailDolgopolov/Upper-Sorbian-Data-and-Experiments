import time

import spacy

from Spacy.my_sentencizer import *

file_name = "../Data/citanka-prose.txt"

start_time = time.time()
nlp = spacy.load("../Spacy/Training/Output/TaggerMorpherParser98/model-best")
nlp.add_pipe("hsb_split", before="tagger")