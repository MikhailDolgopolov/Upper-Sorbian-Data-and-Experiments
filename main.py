import spacy
import pandas as pd
import tqdm

from spacy_conllu import ConlluParser, init_parser

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from Spacy.utils import draw_deps, spacify_text, write_doc_to_conllu, render_pd_table, get_pd_table, spacify_text_file

t1 = 'Dale dyrbi so rjec, zo so w třoch wjesnych hatach stajnje wulka syła husow kupaše.'

t2 = "Tak bu Kocor tež sobuzałožeŕ serbskich spěwanskich swjedźenjow l. 1845 a sćěhowacych, a z tym njesmjertny zbudźowaŕ noweje serbskosće."
doc = ConlluParser(init_parser("hsb")).parse_conllu_file_as_spacy("Data/hsb_UD.conllu", input_encoding="utf-8", combine=True, pair=False)

print(len(list(doc.sents)))

print(len([t.text for t in doc if t.is_alpha]))