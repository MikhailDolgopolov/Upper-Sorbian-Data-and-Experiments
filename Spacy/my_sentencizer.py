import string
from typing import Callable, List, Optional

import srsly
from spacy import Language, util
from spacy.lang import hsb
from spacy.pipeline import Pipe
from spacy.pipeline.senter import senter_score
from spacy.tokens import Doc

@Language.component("hsb_split")
def custom_sentencizer(doc):
    for i, token in enumerate(doc[:-2]):
        # print(token, len(token))
        if token.text.strip() in ".!?":
            if doc[i+2].text not in ".!?":
                doc[i + 1].is_sent_start = True
        else:
            # # Explicitly set sentence start to False otherwise, to tell
            # # the parser to leave those tokens alone
            doc[i + 1].is_sent_start = False
    return doc