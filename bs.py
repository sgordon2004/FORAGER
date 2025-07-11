"""
This module serves as the BS detector. It splits the raw LLM claim into atomic claims and compares them to evidence.
spaCy is an open-source Python library for natural language processing (NLP). It handles tasks like:
    - sentence segmentation
    - part-of-speech (POS) tagging
    - named entity recognition (NER)
    - dependency parsing (how words relate gramatically)
    - Lemmatization (base forms of words)
FORAGER uses spaCY to extract subject-verb-object_ (SVO) relationships from sentences, which helps isolate claims.

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
    - This is the DIRECT object_ of the verb.
So, the object_ is 'market'. But there are also modifiers or market:
    - 3DIC is `compound` -> describes 'market'
    - 'the' is `det` -> determiner for 'market'
So we build the full object_ as "the 3DIC market"

Thus, the final assembled claim is:
    Subject: TSMC
    Verb: dominates
    object_: the 3DIC market

"""

import spacy
# Load pretrained language model
nlp = spacy.load("en_core_web_sm")

doc = nlp("TSMC dominates the 3DIC market.")

# List of discourse markers to skip during extraction
DISCOURSE_MARKERS = {"however", "although", "though", "meanwhile",
                     "nevertheless", "nonetheless", "in contrast", "on the other hand"}

def clean_subtree(subtree):
    return " ".join([
        t.text for t in subtree
        if t.text.lower() not in DISCOURSE_MARKERS
    ])

def extract_atomic_claims(text):
    """
    This method extracts individual, atomic claims from a passage.

    Args:
        text (str): The text from which to extract claims.
    
    Returns:
        claims (list):  A list of atomic claims from the text.
    """
    doc = nlp(text) # Convert text to annnotated spaCy doc
    claims = [] # Initialize empty list to hold all claims in text

    # doc.sents is an iterator over the sentences in a processed document
    for sent in doc.sents:

        # Skip any non-assertive statements by looking for hedging words
        if any(modal in sent.text.lower() for modal in ["might", "could", "if", "possibly", "maybe"]):
            continue

        subject, verb, object_ = None, None, None
        passive_subject, agent = None, None

        # Iterate through each token in the sentence
        for token in sent:
            # Identify the nominal subject
            if token.dep_ == "nsubj":
                # Grab entire phrase representing subject, not just the subject itself
                subject = clean_subtree(token.subtree)
            # Identify nominal subjects in the passive voice
            elif token.dep_ == "nsubjpass":
                passive_subject = clean_subtree(token.subtree)
            elif token.dep_ == "ROOT":
                verb = token.lemma_

                # Find "by" + its object anywhere under the ROOT token
                # Debugging statements
                # print("\n[DEBUG] Verb subtree:")
                for descendant in token.subtree: # look at all the children of the ROOT (verb)
                    # print(descendant.text, descendant.dep_, descendant.head.text)
                    if descendant.dep_ in ("prep", "agent") and descendant.text.lower() == "by":
                        for child in descendant.children: # Look at all the grandchildren of the ROOT (verb)
                            if child.dep_ == "pobj":
                                agent = clean_subtree(child.subtree)
            if token.dep_ == "xcomp":
                object_ = clean_subtree(token.subtree)
            elif token.dep_ in ("dobj", "attr", "pobj") and object_ is None:
                object_ = clean_subtree(token.subtree)
        
        # # Debugging statements
        # print("---")
        # print("SENTENCE:", sent.text)
        # print("PASSIVE_SUBJECT:", passive_subject)
        # print("AGENT:", agent)
        # print("VERB:", verb)

        # Active voice
        if subject and verb and object_:
            claim = f"{subject} {verb} {object_}"
            claims.append(claim)
        # Passive voice -> rewrite to active
        elif passive_subject and agent and verb:
            claim = f"{agent} {verb} {passive_subject}"
            claims.append(claim)

    return claims

text_1 = "TSMC is the global leader in semiconductor manufacturing. It produces chips for major firms like Apple and AMD. The company is headquartered in Hsinchu, Taiwan. Its revenue exceeded $70 billion in 2023. TSMC's advanced 3nm process began mass production in early 2023. If geopolitical tensions increase, the company might diversify its manufacturing locations."
text_2 = "Intel has invested billions in next-generation packaging technology. Its Foveros and EMIB technologies aim to enhance performance and reduce power consumption. Although it trails TSMC in overall market share, Intel plans to compete aggressively in advanced nodes. The company is based in Santa Clara, California. It might regain leadership by 2027 if its roadmap stays on track."
text_3 = "Research suggests that chiplet-based architectures improve performance per watt. AMD’s Ryzen processors use chiplets to separate compute and I/O functions. NVIDIA, on the other hand, focuses heavily on monolithic designs. If yields improve, more companies could shift to chiplet-based strategies. Some engineers argue that chiplets introduce interconnect complexity."
text_4 = "Apple designs its own chips using ARM architecture. These chips are manufactured by TSMC using cutting-edge nodes. Qualcomm and MediaTek also rely on TSMC for fabrication. In contrast, Intel manufactures most of its chips in-house. While Samsung produces both memory and logic chips, it lags behind TSMC in foundry services. Analysts believe that AI workloads will drive demand for 2.5D and 3D packaging."
text_5 = "TSMC might open a new fab in Germany. Some speculate that geopolitical pressures could accelerate this move. If subsidies are approved, the project will likely begin in 2026. However, no official confirmation has been released. The company declined to comment on its plans."

print(extract_atomic_claims(text_1))
print(extract_atomic_claims(text_2))
print(extract_atomic_claims(text_3))
print(extract_atomic_claims(text_4))
print(extract_atomic_claims(text_5))