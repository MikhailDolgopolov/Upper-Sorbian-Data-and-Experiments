from io import open
from conllu import parse_incr, parse


data_file = open("../Data/hsb_UD.conllu", "r", encoding="utf-8").read()
sentences = parse(data_file)
lookup = dict()
for sent in sentences:
    for token in sent:
        form, lemma = token["form"], token["lemma"]
        if any(c.isalpha() for c in form) and form.lower() != lemma.lower():
            lookup[token["form"].lower()] = token["lemma"].lower()
            lookup[token["form"].capitalize()] = token["lemma"].capitalize()
print(len(lookup))
# print(lookup)

import json
with open('../Data/hsb_lookup.json', 'w',encoding='utf8') as fp:
    json.dump(lookup, fp, ensure_ascii=False, indent=4, sort_keys=True)