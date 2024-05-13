import spacy

# Create a blank English nlp object
nlp = spacy.blank("hsb")
doc = nlp("Běštaj nan a mać, a taj měještaj dźowčičku. Tuta pak běše jara rjana a meješe złoty měsačk na čole, a tohodla měješe čoło přeco zawjazane.")
print("Index:   ", [token.i for token in doc])
print("Text:    ", [token.text for token in doc])
print("Lemmas:    ", [token.lemma_ for token in doc])

print("is_alpha:", [token.is_alpha for token in doc])
print("is_punct:", [token.is_punct for token in doc])
print("like_num:", [token.like_num for token in doc])