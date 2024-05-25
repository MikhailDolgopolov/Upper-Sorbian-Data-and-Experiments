import pathlib
import time
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

from Spacy.my_sentencizer import *

file_name = "../Data/citanka-prose.txt"

start_time = time.time()
nlp = spacy.load("../Spacy/Training/Output/TaggerMorpherParser98/model-best")
nlp.add_pipe("hsb_split", before="tagger")

# doc = nlp('''Hišće Serbstwo njezhubjene,
# swój škit we nas ma,
# nowy duch wšo wosłabjene
# sylnje pozběha:
# Bóh je z nami, wjedźe nas,
# njepřecel so hižom hori,
# Serbjo, Serbjo wostanu, Serbjo dobudu!
#
# Jeno złósć so na nas měri,
# by nas póžrěła,
# njech pak zawistna so šćěri,
# njech so přisłodźa:
# Bóh je z nami, wjedźe nas,
# njeprecel so hižom hori,
# Serbjo, Sierbjo wostanu.
# Serbjo dobudu!''')

print(time.time()-start_time)
read_time = time.time()
docs = []
with open(file_name, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for l in tqdm(lines):
        docs.append(nlp(l))

all = Doc.from_docs(docs)
sentences = []
print(len(all))
for s in tqdm(all.sents):
    sentences.append(nlp(s.text))
bin = DocBin(docs=sentences)
bin.to_disk("../Spacy/Training/corpora/prose.spacy")
