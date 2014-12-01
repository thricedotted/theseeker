# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import kenlm

import cPickle as pickle
import numpy as np

from pattern.en import lemma as _lemma
from pattern.en import tokenize

from nltk.corpus import wordnet

from wordnik.swagger import ApiClient
from wordnik.WordApi import WordApi
from urllib2 import HTTPError

import time

# private things that i don't want on github!!
from config import *

# language model helpers 

lm = kenlm.LanguageModel(KENLM_PATH)

def score_sentence(string, normalized=False):
    if not normalized:
        return lm.score(' '.join(tokenize(string)))
    else:
        # XXX: this "normalization" is actually bullshit, ok
        return lm.score(' '.join(tokenize(string))) / max((len(string)-2), 2)

# similarity helper

#vectors = {}
with open('./vectors.pkl') as f:
    vectors = pickle.load(f)
def word_similarity(word_1, word_2):
    try:
        return np.dot(vectors[word_1], vectors[word_2])
    except KeyError:
        return 0


# wordnik helpers

wordnik_api_key = WORDNICK_API_KEY
wordnik_client = ApiClient(wordnik_api_key, 'http://api.wordnik.com/v4')
word_api = WordApi(wordnik_client)

wordnik_pos_cache = {}
wordnet_pos_cache = {}
def wordnik_pos(word, force_wordnik=False):
    from socket import timeout

    try:

        if word in wordnik_pos_cache:
            print('found {} in wordnik cache'.format(word))
            return wordnik_pos_cache[word]

        if not force_wordnik:
            if word in wordnet_pos_cache:
                print('found {} in wordnet cache'.format(word))
                return wordnet_pos_cache[word]

            syns = wordnet.synsets(word)
            if len(syns) > 0:
                print("wordnet hit for {}".format(word))
                pos = [s.pos() for s in syns][:5]

                pos = ['noun' if p == 'n' 
                            else 'verb' if p == 'v' \
                            else 'adjective' if p == 's' \
                            else 'adverb' if p == 'r' 
                            else ''
                        for p in pos]
                
                wordnet_pos_cache[word] = pos
                return pos

        # do this if above fails
        print('looking up {} on wordnik'.format(word))

        try:
            defs = word_api.getDefinitions(word)

            # maybe if i force a sleep here, it'll stop freaking out
            time.sleep(1)

            try:
                wordnik_pos_cache[word] = [d.partOfSpeech for d in defs if d is not None]

            except TypeError:
                wordnik_pos_cache[word] = []

            return wordnik_pos_cache[word]

        except HTTPError:
            print('!!! http error')
            return []

        except timeout:
            print('!!! timed out on {} !!!'.format(word))
            return []

        return []
    
    except UnicodeEncodeError:
        return []


def is_transitive(word):
    syns = wordnet.synsets(word, pos='v')[:1]
    if len(syns) > 0:
        print('found transitivity in wordnet')
        frames = [id for s in syns for id in s.frame_ids()]
        if any(id in frames for id in (8,11)):
            return True
        else:
            return False
    
    top_senses = wordnik_pos(word, force_wordnik=True)[:5]
    return 'verb-transitive' in top_senses

def is_intransitive(word):
    syns = wordnet.synsets(word, pos='v')[:1]
    if len(syns) > 0:
        print('found intransitivity info in wordnet')
        frames = [id for s in syns for id in s.frame_ids()]
        if any(id in frames for id in (1,2,4)):
            return True
        else:
            return False

    top_senses = wordnik_pos(word, force_wordnik=True)[:5]
    return 'verb-intransitive' in top_senses
    


def lemma(string):
    """
    Wrapper around pattern.en's kinda broken lemma function, which doesn't
    have as large of a lexicon as I wish it did.
    """
    string = string.lower()

    lemma_dict = {u"paid": u"pay",
        u"seaweed": u"seaweed",
        u"elves": u"elf",
        u"wolves": u"wolf",
        u"heroes": u"hero",
        u"movie": u"movie",
        u"zombie": u"zombie",
        u"thing": u"thing",
        u"anything": u"anything",
        u"something": u"something",
        u"everything": u"everything",
        u"mentored": u"mentor",
        u"always": u"always",
        u"perhaps": u"perhaps",
        u"programmed": u"program",
        u"fishes": u"fish",
        u"focused": u"focus",
        u"series": u"series",
        u"indices": u"index",
        u"species": u"species",
        u"leam": u"learn",
        u"leaming": u"learn",
        u"leams": u"learn",
        u"leamed": u"learn",
        u"clothes": u"clothes",
        u"nothing": u"nothing"}

    suffixes = ('is', 'us', 'ss')

    if string in lemma_dict: return lemma_dict[string]
    elif any(string.endswith(x) for x in suffixes): return string
    else: return _lemma(string)

def join_helper(words):
    string = ' '.join(words)
    replacements = [(' .', '.'),
    (' ,', ','),
    (' ?', '?'),
    (' !', '!'),
    (' n\'t', 'n\'t'),
    (' \'', '\''),
    ('"', ''),
    ('  ', ' '),
    (' )', ')'),
    ('( ', '('),
    (' - ', '-'),
    (' ;', ';')
    ]
    #(' \'re', '\'re')
    #(' \'s', '\'s')
    #(' \'d', '\'d')

    for orig, new in replacements:
        string = string.replace(orig, new)

    return string
