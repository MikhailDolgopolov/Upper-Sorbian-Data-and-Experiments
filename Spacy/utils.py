import os
import pathlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import spacy
from spacy import displacy
from spacy.attrs import POS, TAG
from spacy.scorer import Scorer
from tqdm import tqdm

from Spacy.my_sentencizer import *

os.environ['best_model'] = 'D:/Programming/Python/UpperSorbian/Spacy/Training/Output/NewTry'
os.environ['models'] = 'D:/Programming/Python/UpperSorbian/Spacy/Training/Output'

def get_hsb_model(name):
    if name is None:
        path = f"{os.getenv('best_model')}/model-best"
    else:
        path = f"{os.getenv('models')}/{name}/model-best"
    m = spacy.load(path)
    # warn = "Warning! Manually adding sentencizer" +(f" to '{name}'" if name else '')
    # print(warn)
    # m.add_pipe("hsb_split", first=True)
    return m
def spacify_text(text: str, model: str = None) -> Doc:
    text = text.replace("\n", " ")
    try:
        model = get_hsb_model(model)
    except IOError:
        model = spacy.load(model)
    return model(text)

def split_doc(doc: Doc)->list[Doc]:
    nlp = spacy.blank("hsb", vocab=doc.vocab)
    nlp.add_pipe("hsb_split", "senter", first=True)

    ds=[]
    for s in doc.sents:
        ds.append(nlp(s.text))
    return ds

def spacify_text_file(file: str, model: str = None) -> Doc:
    return spacify_text(pathlib.Path(file).read_text(encoding="utf-8"), model)

def write_doc_to_conllu(doc: Doc, path: str, encoding: str = "utf-8", par=1):
    if pathlib.Path(path).exists():
        file = open(path, "a", encoding=encoding)
    else:
        file = open(path, "w", encoding=encoding)
    file.write(f"new par = {par}")
    for s in doc.sents:

        id = f"#sent id = p{par}s{s.id}\n"
        file.write(id)
        file.write(f"#text = {s.text}\n")

        for token in s:
            #         text         lemma          UPOS     XPOS      morph         head         DepRel    Deps
            attrs = [token.text, token.lemma_, token.pos_, None, token.morph, token.head.i + 1, token.dep_, None]
            for i in range(len(attrs)):
                if not attrs[i]:
                    attrs[i] = "_"
                attrs[i] = str(attrs[i])

            attr_str = '\t'.join(attrs)
            line = f"{token.i + 1}\t{attr_str}\n"
            file.write(line)

        file.write("\n")

    file.close()

def write_docs_to_conllu(docs: list[Doc], path: str, encoding: str = "utf-8"):
    open(path, "w").close()
    for i in range(len(docs)):
        write_doc_to_conllu(docs[i], path, encoding, par=i + 1)


def draw_deps(text: str | Doc, output_path: str, format:str="svg", model: str = None):
    if isinstance(text, Doc):
        doc = text
        text = doc.text
    else:
        doc = spacify_text(text, model)

    k = 1
    svgs, paths = [], []
    for s in doc.sents:
        svgs.append(displacy.render(s, style="dep"))
        if not re.search(r"\.svg$", output_path):
            file_name = '-'.join([w.text for w in s[:min(len(doc), 6)] if not w.is_punct])
            final_path = output_path +"/"+ file_name
            paths.append(final_path)
        else:
            m_path = re.search(r"^(.*)\.svg", output_path).group(1)
            paths.append(m_path)
    for i in range(k):
        if k > 1:
            paths[i] += f"{i + 1}"

        Path(paths[i] + ".svg").open("w", encoding="utf-8").write(svgs[i])

    if format!="svg":
        import fitz
        from svglib import svglib
        from reportlab.graphics import renderPDF
        for i in range(k):


            # Convert svg to pdf in memory with svglib+reportlab
            # directly rendering to png does not support transparency nor scaling
            drawing = svglib.svg2rlg(paths[i]+".svg")
            pdf = renderPDF.drawToString(drawing)

            # Open pdf with fitz (pyMuPdf) to convert to PNG
            doc = fitz.Document(stream=pdf)
            pix = doc.load_page(0).get_pixmap(alpha=False, dpi=190)
            pix.save(f"{paths[i]}.{format}")
            os.remove(paths[i]+".svg")

def get_pd_table(doc:Doc, include_punctuation=False):

    words = [token.text for token in doc if not token.is_punct or include_punctuation]
    pos = [token.pos_ for token in doc if not token.is_punct or include_punctuation]
    morph = [str(token.morph) for token in doc if not token.is_punct or include_punctuation]

    df = pd.DataFrame.from_dict({"words":words, "pos":pos}, columns=words, orient="index")

    return df


def render_pd_table(df:pd.DataFrame, output_path:str, format:str="png"):
    latex_str = df.to_latex()
    # print(latex_str)
    import matplotlib.pyplot as plt

    plt.axis("off")
    plt.table(cellText = df.to_numpy(dtype='str'))
    # plt.axis("off")
    # plt.text(0.5, 0.5, latex_str, size=50, ha="center", va="center")
    png_path = f"{output_path}.{format}"

    plt.savefig(png_path, format=format, bbox_inches="tight")
    # plt.close(fig)

