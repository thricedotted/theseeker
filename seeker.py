# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

from pattern.en import conjugate, parsetree, referenced, pluralize

from concept import Concept, ConceptSet
from wikihow import WikiHowArticle, WikiStep

from framemaker import make_page_frame, insert_text, insert_into_page, page_noise

from textutil import join_helper, lemma, \
    is_transitive, is_intransitive, \
    word_similarity, score_sentence, \
    wordnik_pos

import codecs
import glob
import itertools
import numpy as np
import random
import re
import subprocess
import textwrap
import time

def flatten(listOfLists):
    return itertools.chain.from_iterable(listOfLists)

class Seeker(object):
    """
    A Seeker, seeking an answer, or a question, or something in between.
    """

    def __init__(self, concepts=None):
        if concepts is None:
            concepts = ConceptSet()

        self.concepts = concepts

        self.explorations = []

        self.logs = []

        self.dreams = []

        self.affirmations = []


    def explore_and_update(self):
        """
        Perform one round of Exploration. Stores the Exploration and a string
        log of the updates performed.
        """
        start_time = time.time()

        log_lines = []

        log_id = "#{0:05d}".format(len(self.explorations) + 1)
        log_lines.append("{0:*^80}".format(""))
        log_lines.append("{0:*^80}".format(" BEGIN ACTIVITY LOG {} ".format(log_id)))
        log_lines.append("{0:*^80}".format(""))

        log_lines.append("")

        article = WikiHowArticle.create_article()

        exp_log = article.exploration_log().splitlines()
        if len(exp_log) > 43:
            exp_log = '\n'.join(exp_log[:43])
            exp_log += "\n\n{:^80}\n".format("--- WARNING: LOG TRUNCATED ---")
        else:
            exp_log = '\n'.join(exp_log)
            exp_log += "\n\n{:^80}\n".format("--- COMPLETE ---")
        log_lines.append(exp_log)

        exploration = Exploration.from_wikihow(article)
        self.explorations.append(exploration)
        
        # update Seeker concepts with Exploration
        concept_updates = 0
        relation_updates = 0

        # add new concepts
        for lemma, concept in exploration.concepts.items():
            if self.concepts.add_concept(concept):
                concept_updates += 1
                relation_updates += len(concept.relations)

            else:
                try:
                    old_concept = self.concepts.get(concept)
                    relation_updates += old_concept.merge_relations(concept)
                except KeyError:
                    pass

        # relation updates in the other direction? lmao
        for lemma, concept in exploration.concepts.items():
            try:
                old_concept = self.concepts.get(concept)
                if old_concept.relations.add_concept(concept):
                    relation_updates += 1
            except KeyError:
                pass

        log_lines.append("")
        for i in range(50-'\n'.join(log_lines).count('\n')):
            if i % 3:
                log_lines.append("")
            else:
                log_lines.append("{:^80}".format("*"))

        log_lines.append("+{0:-^78}+".format("[ ACTIVITY SUMMARY ]"))

        time_string = "Time: {0:10.5f} ms".format((time.time() - start_time) * 1000)
        log_lines.append("| {0:76} |".format(time_string))
        #error_string = "Errors: {}".format(len(exploration.source.errors))
        #log_lines.append("| {0:76} |".format(error_string))

        log_lines.append("|{0:78}|".format(""))

        new_concept_string = "New concepts:  {}".format(concept_updates)
        total_concept_string = "Total concepts:  {}".format(len(self.concepts))
        log_lines.append("| {0:38}".format(new_concept_string) + "{0:38} |".format(total_concept_string))

        new_relation_string = "New relations: {}".format(relation_updates)
        total_relation_string = "Total relations: {}".format(sum(len(c.relations) for c in self.concepts.values()))
        log_lines.append("| {0:38}".format(new_relation_string) + "{0:38} |".format(total_relation_string))

        log_lines.append("+{0:-^78}+".format(""))

        log_lines.append("")

        log_lines.append("{0:*^80}".format(""))
        log_lines.append("{0:*^80}".format(" END ACTIVITY LOG {} ".format(log_id)))
        log_lines.append("{0:*^80}".format(""))

        log = '\n'.join(log_lines)
        self.logs.append(log)

        return log


    def dream(self):
        dream_results = []
        patent_results = []
        other_results = []

        errors = "==> INITIATE DREAM SEQUENCE..."

        def search_sentences(string, file_list):
            grep_args = ['grep', '-i', string]
            raw_results = subprocess.check_output(grep_args + file_list).decode('utf-8')
            results = [tuple(l.split(u":", 1)) for l in raw_results.split(u"\n")[:-1]]
            return results

        while any(len(x) == 0 for x in (dream_results, patent_results, other_results)):
            try:
                #word = random.choice(self.concepts.things.keys())

                word = ""
                concept = None
                while len(word) < 4:
                    concept = self.explorations[-1].concepts.get_random_concept()
                    if concept.type != Concept.DESCRIPTOR:
                        word = concept.lemma

                search = r"\b{}\b".format(word)
                dream_results = search_sentences(search, glob.glob('dreams/*.txt'))
                patent_results = search_sentences(search, glob.glob('patents/*.txt'))
                other_results = search_sentences(search, glob.glob('other/*.txt'))

            except subprocess.CalledProcessError as e:
                errors += ('NO {}...'.format(word.upper()))

        def dreamfuck(results_list, n_sample=10):
            filenames, strings = zip(*results_list)

            n_sample = min(len(strings), n_sample)

            # my files are broken ugh unicode hack fix
            strings = [s.replace('aEURo', '') for s in strings]
            strings = [s.replace('aEUR', '') for s in strings]

            #sampled_strings = random.sample(strings, n_sample)

            # sort by """coherence"""
            sampled_strings = sorted(strings, key=lambda x: score_sentence(x, normalized=True), reverse=True)[:n_sample]

            all_sents = flatten(parsetree(s) for s in sampled_strings)

            # make sure all the sentences we're taking actually contain the
            # word we want
            def filter_fn(sent):
                return word in sent.string.lower() and len(sent) > 4

            sents = filter(filter_fn, all_sents)

            def censored_terms(sent):
                return [w for w in sent
                        if any(w.pos.startswith(x) for x in ('NN', 'JJ', 'RB', 'VB')) \
                        and not any(self.concepts.contains(lemma(w.string), t) 
                                    for t in (Concept.THING, Concept.ACTION, Concept.DESCRIPTOR))
                       ]

            def sentfuck(sent):
                censored = censored_terms(sent)
                words = [w.string if w not in censored
                             else '▓' * len(w.string) if w.pos.startswith('NN')
                             else '▒' * len(w.string) if (w.pos.startswith('JJ') or w.pos.startswith('RB'))
                             else '░' * len(w.string) if w.pos.startswith('VB')
                             else '_' * len(w.string)
                         for w in sent]

                string = join_helper(words)

                # assume 80 characters?
                try:
                    idx = string.lower().index(word.lower())
                    side_len = (80 - (len(word) + 2)) // 2

                    left = "{:.>{len}}".format(string[max(side_len, idx-side_len):idx], len=side_len)
                    right = "{:.<{len}}".format(string[idx+len(word):idx+len(word)+side_len], len=side_len)
                    
                    return left + "[" + word.upper() + "]" + right

                except ValueError:
                    return None

            fucked_sentences = [sentfuck(s) for s in sents]
            censored = [w for s in sents for w in censored_terms(s)]
            return (list(set(s for s in fucked_sentences if s is not None)), censored)
        
        fucked_and_censored = [dreamfuck(dream_results), dreamfuck(patent_results), dreamfuck(other_results)]
        fucked = [x for r in fucked_and_censored for x in r[0]]
        censored = [x for r in fucked_and_censored for x in r[1]]
        fucked = sorted(fucked, key=lambda x: x.count('▓') + x.count('▒') + x.count('░'))#, reverse=True)

        errors += "SYSTEM READY"

        return ([errors] + fucked, self.build_dream(censored, concept), word)
        

    def build_dream(self, wordlist, seed_concept):
        seed = seed_concept.lemma + '-n' if seed_concept.type == Concept.THING else seed_concept.lemma + '-v'
               
        words_and_lemmas = [(w, lemma(w.string)) for w in wordlist if len(w.string) > 3 and w.string.isalpha()]

        noun_blacklist = ['none', 'male', 'female', 'women', 'men', 
                            'hear', 'thru', 'weirder', 'guy', 'mother', 
                            'father', 'daughter', 'brother', 'mama', 
                            'wife', 'thing', 'soooo', 'chan']

        adj_blacklist = ['such', 'much', 'however', 'about', 'most', 'least',
                            'more', 'less', 'else', 'enough', 'sooo' ]

        verb_blacklist = ['men', 'dont', 'left', 'seem']    # STOP SAYING MEN IS A VERB UGH

        adv_blacklist = ['most', 'least', 'kinda', 'quite', 'down', 
                            'there', 'here', 'alot', 'much', 'such', 
                            'more', 'back', 'else', 'very', 'about', 
                            'sooo', 'rather', 'however', 'thus']

        print('getting nouns...')
        nounset = set(w_lemma for w, w_lemma in words_and_lemmas
                            if w.pos.startswith('NN') 
                            and w.pos != 'NNP' 
                            and w_lemma not in noun_blacklist
                        )

        print(len(nounset))
        nouns = [w for w in nounset if 'noun' in wordnik_pos(w)]

        print('getting adjs...')
        adjset = list(set(w.string.lower() for w, w_lemma in words_and_lemmas
                            if w.pos.startswith('JJ') 
                            and w.string.lower() not in adj_blacklist
                        ))
        adjs = [w for w in adjset if 'adjective' in wordnik_pos(w)]


        print('getting verbs...')
        verbset = list(set(w_lemma for w, w_lemma in words_and_lemmas 
                            if w.pos.startswith('VB') 
                            and w_lemma not in verb_blacklist
                        ))
        verbs = [w for w in verbset if any(pos is not None and pos.startswith('verb') for pos in wordnik_pos(w))]

        print('getting advs...')
        advset = list(set(w.string.lower() for w, w_lemma in words_and_lemmas
                            if w.pos.startswith('RB') 
                            and w.string.lower() not in adv_blacklist
                        ))
        advs = [w for w in advset if 'adverb' in wordnik_pos(w)]

        #random.shuffle(nouns)
        #random.shuffle(adjs)
        nouns = sorted(nouns, key=lambda x: word_similarity(x+'-n', seed), reverse=True)
        adjs = sorted(adjs, key=lambda x: word_similarity(x+'-a', seed), reverse=True)

        a_n_shortest = min(len(nouns), len(adjs))
        nps = [a_n for a_n in zip(adjs[:a_n_shortest], nouns[:a_n_shortest])] + [('', n) for n in nouns[a_n_shortest:]]

        #random.shuffle(verbs)
        random.shuffle(advs)
        verbs = sorted(verbs, key=lambda x: word_similarity(x+'-v', seed), reverse=True)

        a_v_shortest = min(len(verbs), len(advs))
        vps = [a_v for a_v in zip(advs[:a_v_shortest], verbs[:a_v_shortest])] + [('', v) for v in verbs[a_v_shortest:]]

        # sort by similarity
        nps = sorted(nps, key=lambda x: max(word_similarity(x[0]+'-a', seed),
                                            word_similarity(x[1]+'-n', seed)))

        vps = sorted(vps, key=lambda x: word_similarity(x[1]+'-v', seed))

        #random.shuffle(nps)
        #random.shuffle(vps)

        def make_noun_string(np, plural=False):
            # random chance of removing modifier
            #if random.random() < 0.5:
            #    np[0] == ''

            # common mass nouns

            if np[1] in ['data', 'information', 'children', 'people', 'stuff', 'equipment']:
                return ' '.join(np).strip()

            elif any(np[1].lower().startswith(x) for x in ('every', 'any', 'some')) or np[1] in ('nothing', 'nobody'):
                return np[1]

            quantifiers = ['many', 'few', 'several', 'various', 'multiple', 'fewer', 'more']
            if np[0] in quantifiers:
                return np[0] + ' ' + pluralize(np[1])

            else:
                die_roll = random.random()
                if die_roll < 0.15 or plural:
                    return ' '.join((np[0], pluralize(np[1]))).strip()
                elif die_roll < 0.25:
                    return random.choice(('his', 'her', 'their', 'your')) + ' ' + ' '.join(np).strip()
                elif random.random() < 0.45:
                    return referenced(' '.join(np).strip())
                else:
                    return 'the ' + ' '.join(np).strip()

        def make_verb_string(vp, conj='part'):
            # random chance of removing modifier
            #if random.random() < 0.5:
            #    vp[0] == ''

            verb = conjugate(vp[1], conj)

            if verb == 'thinking':
                verb = 'thinking of'

            if verb == 'arriving':
                verb = 'arriving at'

            if verb == 'coming':
                verb = 'coming from'

            if verb == 'going':
                verb = 'going to'

            return ' '.join((vp[0], verb)).strip() 
        
        def get_transitive_vp():
            vp = vps.pop()
            transitivity = is_transitive(vp[1])
            checked = []

            while not transitivity:
                if transitivity is not None:
                    checked.append(vp)

                vp = vps.pop()

                transitivity = is_transitive(vp[1])

            vps.extend(checked)
            return vp

        def get_intransitive_vp():
            vp = vps.pop()
            intransitivity = is_intransitive(vp[1])
            checked = []

            while not intransitivity:
                if intransitivity is not None:
                    checked.append(vp)

                vp = vps.pop()

                intransitivity = is_intransitive(vp[1])

            vps.extend(checked)
            return vp

        story = []

        while True:
            try:
                case = random.randint(0, 12)
                next_sent = ""
                if case == 0:
                    template = "{noun_string}, {verb_string}"

                    noun_string = make_noun_string(nps.pop())
                    verb_string = make_verb_string(get_intransitive_vp())

                    next_sent = template.format(noun_string=noun_string, verb_string=verb_string)

                elif case == 1:
                    template = "{noun_string} {verb_string}"

                    noun_string = make_noun_string(nps.pop(), plural=True)

                    verb_string = make_verb_string(get_intransitive_vp())

                    next_sent = template.format(noun_string=noun_string, verb_string=verb_string)

                elif case == 2:
                    template = "{} {} and {} {}"

                    two_vp = [make_verb_string(get_transitive_vp()) for _ in range(3)]
                    two_np = [make_noun_string(nps.pop()) for _ in range(3)]

                    next_sent = template.format(two_vp[0], two_np[0], two_vp[1], two_np[1])

                elif case == 3:
                    template = "{verb_string} {noun_string}"

                    #np = nps.pop()
                    #noun_string = ' '.join((np[0], pluralize(np[1]))).strip()
                    noun_string = make_noun_string(nps.pop(), plural=True)

                    verb_string = make_verb_string(get_transitive_vp())

                    next_sent = template.format(noun_string=noun_string, verb_string=verb_string)

                elif 4 <= case <= 8:
                    preps = ('on', 'around', 'in', 'near', 'behind', 'over', 'under', 'like')
                    template = "{} " + random.choice(preps) + " {}"

                    noun_strings = [make_noun_string(nps.pop()) for _ in range(2)]

                    if random.random() < 0.5:
                        next_sent = template.format(*noun_strings)
                    else:
                        verb_string = make_verb_string(get_transitive_vp())
                        next_sent = verb_string + ' ' + template.format(*noun_strings)

                elif case == 9:
                    template = "{} while {}"

                    verb_strings = [make_verb_string(get_intransitive_vp()) for _ in range(2)]

                    next_sent = template.format(*verb_strings)

                elif 10 <= case <= 12:
                    template = "{noun_string1} {verb_string} {noun_string2}"

                    noun_string1 = make_noun_string(nps.pop(), plural=True)
                    noun_string2 = make_noun_string(nps.pop(), plural=True)

                    verb_string = make_verb_string(get_transitive_vp())

                    next_sent = template.format(noun_string1=noun_string1, noun_string2=noun_string2, verb_string=verb_string)


                # move the adverb around
                """
                if random.random() < 0.5 and 'ly' in next_sent and ' and ' not in next_sent:
                    words = next_sent.split()
                    dont_move = ['actually', 'really', 'probably', 'nearly', 'solely']
                    ly_words = [w for w in words if w.endswith('ly') and w[0] not in 'aeiou' and w not in dont_move]
                    if len(ly_words) == 1:
                        ly_word = ly_words[0]
                        words.remove(ly_word)
                        words.append(ly_word)
                        next_sent = ' '.join(words)
                """

                story.append(next_sent)

            except IndexError:
                break

        story = [s for s in story if not any(s.endswith(x) for x in ('to', 'from', 'at', 'of'))]
        
        def sent_heuristic(sentence, normalized=False):
            print(seed)
            words = sentence.replace(',', '').split()
            print(words)

            adj_relevance = [word_similarity(w+'-a', seed) for w in words]
            verb_relevance = [word_similarity(lemma(w)+'-n', seed) for w in words]
            noun_relevance = [word_similarity(lemma(w)+'-v', seed) for w in words]

            adj_relevance = [s for s in adj_relevance if s > 0]
            verb_relevance = [s for s in verb_relevance if s > 0]           
            noun_relevance = [s for s in noun_relevance if s > 0 ]

            total_rels = len(adj_relevance) + len(verb_relevance) + len(noun_relevance)
            #total_rels = (len(adj_relevance) + len(noun_relevance)) if seed.endswith('-n') else (len(verb_relevance) + len(noun_relevance))
            if total_rels == 0:
                total_rels = 1

            relevance = sum(adj_relevance) + sum(verb_relevance) + sum(noun_relevance)
            #relevance = (sum(adj_relevance) + sum(noun_relevance)) if seed.endswith('-n') else (sum(verb_relevance) + sum(noun_relevance))

            print("relevance: {}".format(relevance))

            score = score_sentence(sentence.replace(',', ''), normalized=normalized)
            print("score: {}".format(score))

            interpolation = ((relevance / total_rels) * 0.01) + score
            print("interpolation: {}".format(interpolation))

            return interpolation


        raw_rank = sorted(story, key=lambda x: sent_heuristic(x), reverse=True)[:20]
        for s in raw_rank:
            print(seed, s)
            #raw_input()
        norm_rank = sorted(story, key=lambda x: sent_heuristic(x, normalized=True), reverse=True)[:10]
        for s in norm_rank:
            print(seed, s)
            #raw_input()

        reranked_story = list(set(raw_rank + norm_rank))
        random.shuffle(reranked_story)
        
        original = '. '.join(s.lower() for s in story if all(c.isalpha() or c in ' ,' for c in s)) + '.'
        reranked = '. '.join(s.lower() for s in reranked_story if all(c.isalpha() or c in ' ,' for c in s)) + '.'

        return reranked


    def affirmation(self):
        lines = []

        lines.append( ["WHAT IF", "CONSIDER THAT", "IMAGINE", "PRETEND", "BELIEVE", "KNOW", "FEEL", "I IMAGINE", "I PRETEND", "I BELIEVE", "I KNOW", "I FEEL"] )
        lines.append( ["ONE THING", "EVERYTHING", "NOTHING", "NOT ONE THING", "NOT EVERYTHING"] )
        lines.append( ["IS", "WAS", "WILL BE", "COULD BE", "SHOULD BE", 
                "IS NOT", "WAS NOT", "WILL NOT BE", "COULD NOT BE", "SHOULD NOT BE", "WOULD NOT BE"] )
        lines.append( ["RANDOMIZED", "CHANCE", "ARBITRARY", "UNDIRECTED", "DIRECTED", "DETERMINISTIC", "A RULE", "A CHOICE", "A DECISION", "PREDETERMINED"] )

        final = '\n\n'.join(map(lambda x: ' '.join(l for l in random.choice(x)), lines))
        
        return final




                         




class Exploration(object):
    """
    An Exploration extracts information from an object, such as
    a WikiHowArticle.
    """

    def __init__(self, concepts=None, source=None):
        if concepts is None:
            concepts = ConceptSet()
            
        self.concepts = concepts
        self.source = source

    @classmethod
    def from_wikihow(cls, article):
        """
        Extracts Concept information from a WikiHowArticle.
        """

        things = {}
        actions = {}
        descriptors = {}
        
        for i, step in enumerate(article.steps):
            parse = parsetree(step.main, relations=True)[0]

            new_things = set(Concept(w.string, Concept.THING) for w in parse if w.pos.startswith('NN'))
            new_actions = set(Concept(w.string, Concept.ACTION) for w in parse if w.pos.startswith('VB'))
            new_thing_descriptors = set(Concept(w.string, Concept.DESCRIPTOR) for w in parse if w.pos.startswith('JJ'))
            new_action_descriptors = set(Concept(w.string, Concept.DESCRIPTOR) for w in parse if w.pos.startswith('RB'))

            if len(step.extra) > 0:
                for parse in parsetree(step.extra):
                    more_thing_descriptors = set(Concept(w.string, Concept.DESCRIPTOR) for w in parse if w.pos.startswith('JJ'))
                    new_thing_descriptors.update(more_thing_descriptors)

                    more_action_descriptors = set(Concept(w.string, Concept.DESCRIPTOR) for w in parse if w.pos.startswith('RB'))
                    new_action_descriptors.update(more_action_descriptors)


            for thing in new_things:
                for other_stuff in set.union(new_actions, new_thing_descriptors):
                    if other_stuff.lemma != thing.lemma:
                        thing.add_relation(other_stuff)

                if thing.lemma not in things:
                    things[thing.lemma] = thing
                else:
                    things[thing.lemma].merge_relations(thing)
            
            for action in new_actions:
                for other_stuff in set.union(new_things, new_action_descriptors):
                    if other_stuff.lemma != action.lemma:
                        action.add_relation(other_stuff)

                if action.lemma not in actions:
                    actions[action.lemma] = action
                else:
                    actions[action.lemma].merge_relations(action)

            for descriptor in new_thing_descriptors:
                for thing in new_things:
                    descriptor.add_relation(thing)
                
                if descriptor.lemma not in descriptors:
                    descriptors[descriptor.lemma] = descriptor
                else:
                    descriptors[descriptor.lemma].merge_relations(descriptor)
            
            for descriptor in new_action_descriptors:
                for action in new_actions:
                    descriptor.add_relation(action)

                if descriptor.lemma not in descriptors:
                    descriptors[descriptor.lemma] = descriptor
                else:
                    descriptors[descriptor.lemma].merge_relations(descriptor)

        concepts = ConceptSet(things, actions, descriptors)

        return cls(concepts, article)

#a = WikiHowArticle.create_article()
#e = Exploration.from_wikihow(a)
#seeker = Seeker()

def seeker_to_book():
    #with codecs.open('latex_test.tex', encoding='utf-8') as f:
    #    template = f.read()
    
    seeker = Seeker()

    pages = []

    page_no = 0
    def write_page(text):
        leftover = 60 - text.count('\n') 
        text += '\n' * leftover
        with codecs.open('out/{0:03d}.txt'.format(page_no), 'w', encoding='utf-8') as f:
            f.write(text)

    title = insert_into_page("THE SEEKER\n----------")
    title_lines = title.splitlines()
    title_lines[-4] = "{:^80}".format("BY THRICEDOTTED")
    title_lines[-2] = "{:^80}".format("NANOGENMO 2014")
    title = '\n'.join(title_lines)
    write_page(title)
    page_no += 1

    for i in range(5):
        noise = page_noise(rate=0.05 * i)
        inserted = insert_text(noise, 'this page intentionally left not blank')
        write_page(inserted)
        page_no += 1

    affirmation = seeker.affirmation()
    pages.append(affirmation)
    print(affirmation)
    framed_affirmation = insert_text(make_page_frame(), affirmation)
    write_page(framed_affirmation)
    page_no += 1

    begin = insert_into_page(("... & work & scan & imagine & repeat & ...").strip())
    write_page(begin)
    page_no += 1

    for _ in range(100):
        log = seeker.explore_and_update()
        pages.append(log)
        print(log)
        write_page(log)
        page_no += 1

        raw_dream, censored, concept = seeker.dream()

        # compress dream
        boop = max(min(13, len(raw_dream)-2), 1)
        #print(raw_dream)
        internal = sorted(random.sample(raw_dream[1:-1], boop), 
                          key=lambda x: x.count('▓') + x.count('▒') + x.count('░'))
        dream_end = '==> {} LINES OMITTED...SCAN COMPLETE'.format(len(raw_dream) - 15)
        raw_dream = [raw_dream[0]] + internal + [raw_dream[-1]] + [dream_end]
        dream_text = '\n\n{:^80}\n\n'.format("*").join(raw_dream)
        pages.append(dream_text)
        print(dream_text)
        write_page(dream_text)
        page_no += 1

        # and unvisions
        lines = textwrap.fill(censored, width=50).splitlines()
        if len(lines) > 16:
            lines = lines[:16]

        unvision = ' '.join("unvision : {}".format(concept.lower()))
        unvision += '\n' + '-' * len(unvision) + "\n\n\n"
        unvision += '( ' + '\n\n'.join(lines)

        unvision = unvision.rsplit('.', 1)[0] + '. )'

        unvision += "\n\n\n---"
        print(unvision)
        write_page(insert_into_page(unvision))
        page_no += 1

        if random.random() < 0.2:
            affirmation = seeker.affirmation()

            pages.append(affirmation)
            print(affirmation)

            framed_affirmation = insert_text(make_page_frame(), affirmation)

            write_page(framed_affirmation)
            page_no += 1

        else:
            noise = page_noise(rate=max(random.triangular()-0.2, 0.2))
            write_page(noise)
            page_no += 1

    affirmation = seeker.affirmation()
    pages.append(affirmation)
    print(affirmation)
    framed_affirmation = insert_text(make_page_frame(), affirmation)
    page_no -= 1
    write_page(framed_affirmation)
    page_no += 1

    for i in range(4, 0, -1):
        noise = page_noise(rate=0.05 * i)
        write_page(noise)
        page_no += 1
        
    end = insert_into_page("END")
    write_page(end)
    page_no += 1
