from __future__ import division, unicode_literals

from pattern.en import parsetree, conjugate
from textutil import lemma
from pattern.web import URL, Element, plaintext

import textwrap
import time

class WikiHowArticle(object):
    """
    Structure to hold information scraped from a wikiHow article.
    """

    def __init__(self, url, title=None, steps=None, tips=None, errors=None):
        self.url = url
        self.title = title
        self.steps = steps
        self.tips = tips

        if errors is None: errors = []
        self.errors = errors

    def __repr__(self):
        return u'WikiHowArticle("{}")'.format(self.url)

    def exploration_log(self, type="COMPUTER"):
        """
        Return a string of the exploration log for this article.
        """
        if type == "COMPUTER":

            def step_to_computer_verbs(string):
                parse = parsetree(string, relations=True)[0]

                def convert_words(v):
                    return ' '.join(lemma(w.string) if w.pos.startswith('VB') else w.string
                            for w in v.words 
                            if lemma(w.string) != "be" and w.string.lower() not in ('to', 'wo') and w.string.isalpha())

                chunks = [convert_words(v) for v in parse.verbs]
                
                return ' -- '.join(c for c in chunks if len(c) > 0).upper()

            def step_to_computer_nouns(string):
                parse = parsetree(string, relations=True)[0]

                def convert_words(v):
                    return '_'.join(lemma(w.string)
                            for w in v.words 
                            if w.pos != "DT" and not w.pos.startswith('PRP') and w.string.isalpha()
                            ).replace('-', '_')

                chunks = [convert_words(v) for v in parse.chunk if v.pos == 'NP']
                
                return ' | '.join(c for c in chunks if len(c) > 0).upper()

            def step_to_computer_adjs(string):
                try:
                    parse = parsetree(string, relations=True)[0]
                    return ' '.join(a.string for a in parse.adjectives if a.string.isalpha())
                except IndexError:
                    return ''

            verbs = [step_to_computer_verbs(s.main) for s in self.steps]
            verbs = [v if len(v) > 0 else '???' for v in verbs]

            nouns = [step_to_computer_nouns(s.main) for s in self.steps]
            nouns = [n if len(n) > 0 else '***' for n in nouns]

            adjs = [' '.join(set(' '.join((step_to_computer_adjs(s.main),
                     step_to_computer_adjs(s.extra))).split()))
                     for s in self.steps]
            adjs = [a.lower() if len(a) > 0 else '!null' for a in adjs]

            lines = ['{:02d}... {} ( {} ) => {}'.format(i, v, n, a)
                      for i, v, n, a in zip(range(len(verbs)), verbs, nouns, adjs)]

            title_words = self.title.lower().strip()[7:].split()
            title_words = [w for w in title_words if w not in ("your", "a", "an", "the")]

            if title_words[0].lower() != "not":
                title_words[0] = conjugate(lemma(title_words[0]), 'part').upper()
            else:
                title_words[1] = conjugate(lemma(title_words[1]), 'part').upper()

            title = "{0:^80}".format("- " + ' '.join(title_words) + " -").upper()

            fixed_lines = [textwrap.fill(textwrap.dedent(line).strip(),
                                         initial_indent='', 
                                         subsequent_indent='      ', 
                                         width=76)
                           for line in lines]

            return '\n\n'.join([title] + fixed_lines)

        else:
            raise NotImplementedError("Type {} of logging not implemented!".format(type))

    @classmethod
    def create_article(cls, title=None):
        page = cls.get_raw_wikihow_page(title=title) if title is not None \
               else cls.get_raw_wikihow_page() 

        title = Element(page)("h1.firstHeading a")[0].string
        if title.startswith("wiki"): title = title[4:]

        url = 'http://www.wikihow.com/{}'.format(title[7:].replace(' ', '-'))

        steps, errors = cls.get_steps(page)
        tips = cls.get_tips(page)

        return cls(url, title, steps, tips, errors)

    @staticmethod
    def get_raw_wikihow_page(title=None):
        if title is not None and 'how to' in title.lower():
            title = title.lower().replace('how to', '', 1).strip()

        # keep to "human" articles
        #allowed_cats = ['Youth', 'Family Life', 'Relationships', 'Personal Care and Style', 'Work World']
        allowed_cats = ['Youth', 'Family Life', 'Relationships']
        main_cat = ""
        s = ""

        while main_cat not in allowed_cats:
            try:
                s = URL('http://www.wikihow.com/{}'.format(title)).read() if title is not None \
                    else URL('http://www.wikihow.com/Special:Randomizer').read()
                main_cat = Element(s)('ul#breadcrumb li a')[2].string
                print(main_cat)
            except:
                time.sleep(5)

        return s

    @staticmethod
    def get_steps(page):
        """
        Extract steps from a wikiHow HTML page.
        """
        errors = []
        steps = []

        e = Element(page)

        for s in e('.steps li'):
            try:
                main = s('b.whb')[0].string
                extra = s.string[s.string.index(main) + len(main) + 4:]

                if '<div class="clearall">' in extra:
                    extra = extra[:extra.index('<div class="clearall">')]

                steps.append(WikiStep(plaintext(main), plaintext(extra)))

            except Exception as e:
                errors.append(e)

        return (steps, errors)


    @staticmethod
    def get_tips(page):
        """
        Extract tips from a wikiHow HTML page.
        """
        e = Element(page)

        tips = [plaintext(s.string) for s in e('#tips li')]

        return tips


class WikiStep(object):
    """
    One step from a wikiHow article.
    """
    def __init__(self, main, extra):
        self.main = main
        self.extra = extra

    def __repr__(self):
        return u'WikiStep({}, {})'.format(repr(self.main), repr(self.extra))
