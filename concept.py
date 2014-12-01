from textutil import lemma

import random

class Concept(object):
    # constants used for concept types
    THING = "THING"
    ACTION = "ACTION"
    DESCRIPTOR = "DESCRIPTOR"
    
    """
    Concepts are surface words that the Seeker learns on an Exploration.
    """
    
    def __init__(self, string, type):
        self.lemma = lemma(string)
        self.type = type
        self.relations = ConceptRelationSet()

    def __hash__(self):
        return self.lemma.__hash__()

    def __repr__(self):
        return "Concept({}, {})".format(repr(self.lemma), repr(self.type))

    def add_relation(self, concept):
        """
        Adds a pointer to an existing Concept to this Concept's relations.

        Returns True if the relation was added, False if it was already
        present.
        """
        return self.relations.add_concept(concept)

    def merge_relations(self, concept):
        if self.lemma != concept.lemma:
            raise Exception(u"Lemmas do not match! ({}, {})".format(self.lemma, concept.lemma))

        return self.relations.merge_relations(concept.relations)

    def get_random_relation(self, type=None):
        return self.relations.get_random_relation(type=type)


class ConceptRelationSet(object):
    def __init__(self, things=None, actions=None, descriptors=None):
        if things is None: things = set()
        if actions is None: actions = set()
        if descriptors is None: descriptors = set()

        self._concept_sets = { Concept.THING: things, 
                                Concept.ACTION: actions,
                                Concept.DESCRIPTOR: descriptors
                              }

        self.things = self._concept_sets[Concept.THING]
        self.actions = self._concept_sets[Concept.ACTION]
        self.descriptors = self._concept_sets[Concept.DESCRIPTOR]

    def __len__(self):
        return len(self.things) + len(self.actions) + len(self.descriptors)

    def get_random_relation(self, type=None):
        if type is None:
            return random.choice(list(self.things) + list(self.actions) + list(self.descriptors))
        else:
            return random.choice(list(self._concept_sets[type]))

    def contains(self, string, type):
        try:
            return lemma(string) in self._concept_sets[type]

        except KeyError:
            raise KeyError("Invalid or undefined concept type")

    def add_concept(self, concept):
        try:
            if not self.contains(concept.lemma, concept.type):
                self._concept_sets[concept.type].add(concept.lemma)
                return True
            else:
                return False

        except KeyError:
            raise KeyError("Invalid or undefined concept type")

    def merge_relations(self, other):
        """
        Merges two relation sets.

        Returns number of new relations added to self from other.
        """
        n_updates = 0

        for type in (Concept.THING, Concept.ACTION, Concept.DESCRIPTOR):
            for concept_lemma in other._concept_sets[type]:
                if concept_lemma not in self._concept_sets[type]:
                    self._concept_sets[type].add(concept_lemma)
                    n_updates += 1

        return n_updates


class ConceptSet(object):
    def __init__(self, things=None, actions=None, descriptors=None):
        if things is None: things = {}
        if actions is None: actions = {}
        if descriptors is None: descriptors = {}

        self._concept_dicts = { Concept.THING: things, 
                                Concept.ACTION: actions,
                                Concept.DESCRIPTOR: descriptors
                              }

        # aliases to make my life easier
        self.things = self._concept_dicts[Concept.THING]
        self.actions = self._concept_dicts[Concept.ACTION]
        self.descriptors = self._concept_dicts[Concept.DESCRIPTOR]

    def __len__(self):
        return len(self.things) + len(self.actions) + len(self.descriptors)

    def items(self):
        return self.things.items() + self.actions.items() + self.descriptors.items()

    def values(self):
        return self.things.values() + self.actions.values() + self.descriptors.values()

    def keys(self):
        return self.things.keys() + self.actions.keys() + self.descriptors.keys()

    def add_concept(self, concept):
        try:
            if not self.contains(concept.lemma, concept.type):
                self._concept_dicts[concept.type][concept.lemma] = concept
                return True
            else:
                return False

        except KeyError:
            raise KeyError("Invalid or undefined concept type")

    def contains(self, string, type):
        try:
            return lemma(string) in self._concept_dicts[type]

        except KeyError:
            raise KeyError("Invalid or undefined concept type")
    
    def get(self, concept):
        return self._concept_dicts[concept.type][concept.lemma]

    def get_by_lemma(self, lemma, type):
        return self._concept_dicts[type][lemma]

    def get_random_concept(self, type=None):
        if type is None:
            return random.choice(self.things.values() + self.actions.values() + self.descriptors.values())
        else:
            return random.choice(self._concept_dicts[type].values())
