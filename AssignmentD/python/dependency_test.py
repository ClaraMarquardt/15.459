# -------------------------------------------------
# Dependency tests ----------------------------------
# -------------------------------------------------

## nltk_test

def nltk_test():

	## initialize
	import nltk

	## punkt (tokenizer)
	print(nltk.tokenize.word_tokenize('This is a red giraffe'))

	## wordnet (corpus)
	from nltk.corpus import wordnet as wn
	print(wn.synsets('happiness'))

# ------

## spacy_test

def spacy_test(spacy_dir='en_core_web_sm'):

	# initialize
	import spacy
	nlp = spacy.load(spacy_dir)

	# test
	doc = nlp(u'This is a Spacy test. These are two purple elephants.')
	print([sent.text for sent in doc.sents])
