import spacy

from Spacy.utils import draw_deps, spacify_text, write_doc_to_conllu

t1 = 'Dale dyrbi so rjec, zo so w třoch wjesnych hatach stajnje wulka syła husow kupaše.'

t2 = "Tak bu Kocor tež sobuzałožeŕ serbskich spěwanskich swjedźenjow l. 1845 a sćěhowacych, a z tym njesmjertny zbudźowaŕ noweje serbskosće."
doc = spacify_text(t2)

print(list(doc.sents))

# draw_deps(doc, "Images", format="jpeg")
# write_doc_to_conllu(doc, "test.conllu")


# draw_deps("Now I become death, the destroyer of worlds.", "Images", format="jpeg", model="en_core_web_sm")
