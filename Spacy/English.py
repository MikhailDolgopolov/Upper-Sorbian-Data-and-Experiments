import spacy

nlp = spacy.load("en_core_web_trf")

data = str(open("../Data/eng.txt").read())
print(data)
doc = nlp(data)
for token in doc:
    print(f"{token.text:<20}  {token.lemma_:<10} {token.pos_:<10} {token.morph}")