import spacy
import torch
print(torch.cuda.is_available())

output_dir="Models/MyMorph"
# main(output_dir)
nlp2 = spacy.load(output_dir)
test_text="Tak so mi nihdźe, nihdźe njelubi, kaz mjez mojimi Serbami. "
doc = nlp2(test_text)
print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])