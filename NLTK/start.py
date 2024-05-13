import nltk
# nltk.download('punkt')
# sentence = """At eight o'clock on Thursday morning
# ... Arthur didn't feel very good."""
# tokens = nltk.word_tokenize(sentence, "english")
# print(tokens)

import polyglot
from polyglot.downloader import downloader
from polyglot.text import Text, Word

text = Text("Běštaj nan a mać, a taj měještaj dźowčičku. Tuta pak běše jara rjana a meješe złoty měsačk na čole, a tohodla měješe čoło přeco zawjazane.", hint_language_code="hsb")
print(downloader.supported_tasks(lang="hsb"))
for w in text.words:
  w = Word(w, language="hsb")
  print(f"{w:<20}{w.morphemes}")