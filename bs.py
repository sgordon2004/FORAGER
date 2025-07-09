"""
This module serves as the BS detector. It splits the raw LLM claim into atomic claims and compares them to evidence.
spaCy is an open-source Python library for natural language processing (NLP). It handles tasks like:
    - sentence segmentation
    - part-of-speech (POS) tagging
    - named entity recognition (NER)
    - dependency parsing (how words relate gramatically)
    - Lemmatization (base forms of words)
FORAGER uses spaCY to extract subject-verb-object (SVO) relationships from sentences, which helps isolate claims.

en_core_web_sm is one of spaCY's pretrained language models.
    - en = English
    - core = Core model (general-purpose)
    - sm = Small version (lightweight + fast)
When given text, en_core_web_sm returns a fully annotated doc with tokens, sentence boundaries, POS tags, dependencies, etc.

Example:
[python]
doc = nlp("TSMC dominates the 3DIC market.")

for token in doc:
    print(token.text, token.dep_, token.head.text, token.pos_)

[output]
[token.text]    [token.dep_ ]   [token.head.text]   [token.pos_]
TSMC            nsubj           dominates           PROPN
dominates       ROOT            dominates           VERB
the             det             market              DET
3DIC            compound        market              PROPN
market          dobj            dominates           NOUN
.               punct           dominates           PUNCT

How to read the output:
1. Look for dep_ == 'nsubj', which is the nominal subject:
    - That's 'TSMC', which DEPENDS ON 'dominates' (token.head.text)
So, the subject is 'TSMC' and the verb is 'dominates' (the head of the subject)

2. Look for dep_ == 'dobj':
    - That's 'market', which also depensd on 'dominates'
    - This is the DIRECT OBJECT of the verb.
So, the object is 'market'. But there are also modifiers or market:
    - 3DIC is `compound` -> describes 'market'
    - 'the' is `det` -> determiner for 'market'
So we build the full object as "the 3DIC market"

Thus, the final assembled claim is:
    Subject: TSMC
    Verb: dominates
    Object: the 3DIC market

"""

import spacy
# Load pretrained language model
nlp = spacy.load("en_core_web_sm")

doc = nlp("TSMC dominates the 3DIC market.")

for token in doc:
    print(token.text, token.dep_, token.head.text, token.pos_)