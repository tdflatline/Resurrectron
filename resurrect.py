#!/usr/bin/python

# Resurrecter: Takes a soul file and animates it.

try:
  import psyco
  psyco.full()
except: pass

import nltk
import re
import pickle
import random
import math
import numpy
import curses.ascii
#import twitter


# Maybe pos-tag isn't that important for searching. It seems
# to make some things harder... We probably should strip tenses from
# verbs, at least
class POSTrim:
   def __init__(self):
    self.pos_map = { "VBD":"VB", "VBG":"VB", "VBN":"VB", "VBP":"VB", "VBZ":"VB",
                       "JJR":"JJ", "JJS":"JJ",
                       "NNS":"NN", "NNPS":"NN", "NNP":"NN" }

   def trim(self, pos):
     if pos in self.pos_map:
       return self.pos_map[pos]
     else:
       return pos

# RP: particle
# CC: conjunction, coordinating
# MD: modal auxiliary
# PDT: pre-determiner
# POS: genitive marker
# TO: "to" as preposition or infinitive marker
# WDT: WH-determiner (that)
# WP: WH-pronoun (that)
# WP$: WH-pronoun, possessive (whose)
# WRB: Wh-adverb (how, why)
class QueryStripper:
 pass

#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
class PronounInverter:
  def __init__(self):
    self.pronoun_21_map = { "you":"i", "yours":"mine","your":"my" }
    self.pronoun_12_map = { "i":"you", "me":"you", "mine":"yours","my":"your" }

  # XXX: use (?i) instead of this casehack nonsense
  def make_2nd_person(self, string):
    for i in self.pronoun_12_map.iterkeys():
      chars = list(i)
      casehack = ""
      for c in chars: casehack += "["+c.lower()+c.upper()+"]"
      string = re.sub("([^a-zA-Z]|^)"+casehack+"([^a-zA-Z]|$)",
                      "\\1"+self.pronoun_12_map[i]+"\\2", string)
    return string

  def make_1st_person(self, string):
    for i in self.pronoun_21_map.iterkeys():
      chars = list(i)
      casehack = ""
      for c in chars: casehack += "["+c.lower()+c.upper()+"]"
      string = re.sub("([^a-zA-Z]|^)"+casehack+"([^a-zA-Z]|$)",
                      "\\1"+self.pronoun_21_map[i]+"\\2", string)
    return string

  def invert_all(self, string):
    split_string = string.split()
    new_string = []
    for s in split_string:
      snew = self.make_1st_person(s)
      if snew == s:
        snew = self.make_2nd_person(s)
      new_string.append(snew)
    return " ".join(new_string)


# FIXME: Should these be static classes? Some maybe should
# do some record keeping on what they modify for inversion..
postrim = POSTrim()
porter = nltk.PorterStemmer()
proverter = PronounInverter()

#- Notion of "pronoun context" per person
#  - Store list of recent nouns+verbs used in queries/responses
#    - If no specific noun/verb is found, use these for query
#  - Expire after a while
class ConversationContext:
  pass

#- Knowledge Base:
#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
#  - Tag with sender as hidden text, add to list of tweets for searching
class KnowledgeBase:
  pass

#- foursquare
#  - special place query parsing
#  - Random lamentations about favorite places and being trapped in the
#    Internet
#    - Trim down location to key "important" word using tf-idf
#  - Named entity recognition and relation extraction:
#    - http://nltk.googlecode.com/svn/trunk/doc/book/ch07.html
class FourSquareData:
  pass

#- url simmilarity
#  - When given a url, give a similar one back
#  - Or add related urls to generated sentences.
#  - summarize the url
#    - http://stackoverflow.com/questions/626754/how-to-find-out-the-summarized-text-of-a-given-url-in-python-django
#      - http://tristanhavelick.com/summarize.zip
#    - feed summary into SearchableTextCollection
#      - score summary sentences based on similarity to bot's word frequency
#        tables..
class URLClassifier:
  pass

class TagBin:
  def __init__(self, text, tags):
    # XXX: Denormalize to get orig text?
    # XXX: Normalize tags?
    self.tags = tags
    self.orig_text = text
    self.tagged_tokens = [nltk.tag.util.tuple2str((porter.stem(t[0]).lower(),
                            postrim.trim(t[1]))) for t in self.tags]
    self.vocab = set(self.tagged_tokens)

# XXX: Remove all tags
class SearchableText(TagBin):
  skip_tokens = set(['"','(',')','[',']'])
  def __init__(self, text, hidden_text=""):
    if hidden_text:
      hidden_text = hidden_text.rstrip()
      if not curses.ascii.ispunct(hidden_text[-1]):
        hidden_text += ". "
      else:
        hidden_text += " "
    self.hidden_text = hidden_text

    if not curses.ascii.ispunct(text[-1]): text += "."

    # Hidden text will ruin our markov model. Need to avoid it for training
    tokens = []
    for w in nltk.word_tokenize(hidden_text+text):
      if w: tokens.append(w)
    if not tokens: print "Empty text!"
    tags = nltk.pos_tag(tokens)
    TagBin.__init__(self, text, tags)

  def count(self, word):
    return self.tagged_tokens.count(word)


# Search:
#  1. Tag, tolower(), stem
#  2. if strip: strip all but nouns, adj, verbs
#  3. tf-idf to score text record from collection
#  4. TODO: If score is 0, requery for "WTF buh uh huh dunno what talking about"
#  5. FIXME: Return probabilistically based on score with some cutoff
#  6. TODO: try to avoid replying to questions with questions. score lower.

# TODO: Build the score vectors using both the tagged and untaged words
# Then if tags and words match, you get more score than just tags,
# but its not total fail if your question can't be parsed..
class SearchableTextCollection:
  def __init__(self, texts=None):
    self._idf_cache = {}
    self.texts = texts
    if texts:
      self.needs_update = True
      self.update_matrix()

  def add_text(self, text, update=False):
    self.texts.append(text)
    self.needs_update = True
    if update: self.update_matrix()

  def update_matrix(self):
    print "Building Terms..."
    terms = set()
    for t in self.texts:
      terms.update(t.vocab)
    self.vocab = list(terms)
    self.vocab.sort()
    print "Built Terms."
    self.D = []
    for doc in self.texts:
      d = []
      for dt in self.vocab:
        if dt in doc.vocab: d.append(self.tf_idf(dt, doc))
        else: d.append(0.0)
      d = numpy.array(d)
      d /= math.sqrt(numpy.dot(d,d))
      self.D.append(d)
    print "Computed score matrix."
    self.needs_update = False

  def query(self, query_string, randomize=False, only_pos=[]): #only_pos=["FW", "CD", "$", "N.*"]):
    if self.needs_update: self.update_matrix()
    print "Building Qvector.."

    query_string = proverter.invert_all(query_string)
    print "Inverted Query: "+query_string
    query_text = SearchableText(query_string)

    # XXX: Def strip off WH*, DT* TO* w/ a querystripper
    # XXX: Hrmm, could amplify nouns and dampen adjectives and verbs..
    # How though? Normalize to 0.75*(max_noun/max_word)
    # Insert a You/NN in all queries?
    # FIXME: If no nouns, only pronouns, use state from queries+responses
    q = []
    for dt in self.vocab:
      if dt in query_text.vocab:
        ok = not bool(only_pos)
        for p in only_pos:
          if re.match("/"+p, dt): ok = True
        if ok: q.append(self.tf_idf(dt, query_text))
        else: q.append(0.0)
      else: q.append(0.0)

    q = numpy.array(q)
    q /= math.sqrt(numpy.dot(q,q))

    print "Qvector built"
    scores = []
    tot_score = 0.0
    for d in self.D:
      score = numpy.dot(q,d)
      tot_score += score
      scores.append(score)

    print "Matrix multiplied"

    # FIXME: hrmm. This will mess with our pronoun logic
    if tot_score == 0.0:
      print "Zero score.. Recursing"
      return self.query(SearchableText("WTF buh uh huh dunno what talking about understand"),
                        randomize, only_pos)

    if randomize:
      sorted_scores = []
      for i in xrange(len(scores)):
        sorted_scores.append((scores[i], i))
      sorted_scores.sort(lambda x,y: int(y[0]*10000 - x[0]*10000))

      # Choose from top 5 scores..
      # FIXME: This is still kind of arbitrary
      top_quart = 0
      count = 0.0
      for i in xrange(5):
        top_quart += sorted_scores[i][0]
      choice = random.uniform(0, top_quart)
      for i in xrange(5):
        count += sorted_scores[i][0]
        if count >= choice:
          print "Rand score: "+str(sorted_scores[i][0])+"/"+str(sorted_scores[0][0])
          print "Choice: "+str(choice)+" count: "+str(count)+" top_quater: "+str(top_quart)
          return self.texts[sorted_scores[i][1]]
      print "WTF? No doc found: "+str(count)+" "+str(tot_score)
      return random.choice(self.texts)
    else:
      max_idx = 0
      for i in xrange(len(scores)):
        if scores[i] > scores[max_idx]:
          max_idx = i
      print "Max score: "+str(scores[max_idx])
      return self.texts[max_idx]

  def tf(self, term, text):
    """ The frequency of the term in text. """
    return float(text.count(term)) / len(text.tagged_tokens)

  def idf(self, term):
    """ The number of texts in the corpus divided by the
    number of texts that the term appears in. 
    If a term does not appear in the corpus, 0.0 is returned. """
    # idf values are cached for performance.
    idf = self._idf_cache.get(term)
    if idf is None: 
      matches = len(list(True for text in self.texts if term in text.vocab))
      if not matches:
        idf = 0.0
      else:
        idf = math.log(float(len(self.texts)) / matches)
      self._idf_cache[term] = idf
    return idf

  def tf_idf(self, term, text):
    return self.tf(term, text) * self.idf(term)

# Mutate:
#  1. For each noun, adj and verb not in query:
#     random choice:
#       A. nltk.text.Text.similar(w) from corups
#          Replace with tf-idf weighted word
#       C. tf-idf score wordnet synonym set, choose based on that
#       B. tf-idf score: http://labs.google.com/sets
class PhraseMutator:
  pass


def main():
  pass
  #soul = SoulCorpus('target_user')
  #pickle.dump(soul, open("target_user.soul", "w"))
  #soul = pickle.load(open("target_user.soul", "r"))
  #pg = PhraseGenerator(soul)

  #pg.test_run()
  #return


  #soul.tweet_collection.generate(100)

if __name__ == "__main__":
  main()
