#!/usr/bin/python
import nltk
import re

import AGFL
from tokenizer import word_tokenize, word_detokenize


# These tend to be pretty solid.
class AGFLTweaker:
  nltk_ep4_map = {
    "PRP$" : "POSSPRON", # pronoun, possessive
    "RB" : "ADVB(modf)", # adverb
    "RBR" : "ADVB(comp)", # adverb, comparative
    "RBS" : "ADVB(supl)", # adverb, superlative
    "RP" : "PARTICLE(none)", # particle. XXX: these next 4 should have the word?
    "JJ" : "ADJE(abso,none)", # adjective or numeral, ordinal 
    "JJR" : "ADJE(comp,none)", # adjective, comparative
    "JJS" : "ADJE(supl,none)", # adjective, superlative
    "CC" : "CON(coo)", # conjunction, coordinating
    "IN" : "CON(sub)", # preposition or conjunction, subordinating
    "UH" : "INTERJ", # interjection
    "CD" : "NUM(card)", # numeral, cardinal
    "." : "-."
  }

  # These tags tend not to work very well with NLTK.
  # Leave them (and others) as their NLTK equivs to
  # let the HMM compensate a bit using context. Left here
  # for reference.
  nltk_ep4_map_junky = {
    # NLTK is too generous with tagging things as nouns
    "NN" : "NOUN(sing)", # noun, common, singular or mass
    "NNP" : "NOUN(sing)", # noun, proper, singular
    "NNPS" : "NOUN(plur)", # noun, proper, plural
    "NNS" : "NOUN(plur)", # noun, common, plural

    # These need some review..
    "VB" : "VERBI(NONE,none,cplx)", # verb, base form
    "VBD" : "VERBP(NONE,none,cplx)", # verb, past tense
    "VBG" : "VERBG(NONE,none,cplx)", # verb, present participle or gerund
    "VBN" : "VERBP(NONE,none,trav)", # verb, past participle
    "VBP" : "VERBV(NONE,none,cplx)", # verb, present tense, not 3rd person singular
    "VBZ" : "VERBS(None,none,cplx)",# verb, present tense, 3rd person singular
    "PRP" : "PERSPRON(nltk,nltk,nltk)" # pronoun, personal
  }

  # Temporarily strip out stuff we left in (esp for AGFL)
  def agfl_fix(self, tokens, nltk_tags):
    fixed = False
    for t in xrange(len(nltk_tags)):
      tokens[t] = re.sub(r"\.\.[\.]+", "...", tokens[t])
      if nltk_tags[t][0] == "'s":
        # FIXME: Add to normalizer/undo?
        if nltk_tags[t][1] == "VBZ":
          tokens[t] = "is"
        elif nltk_tags[t][1] == "POS": # Evil Hack. XXX: Undo?
          tokens[t-1] += "s"
          nltk_tags[t-1] = (nltk_tags[t-1][0]+"s", nltk_tags[t-1][1])
          fixed = True
    if fixed:
      nltk_tags.remove(("'s", "POS"))
      tokens.remove("'s")

  bad_affix = ["^[\#\@\']", "[\']$"]
  replace = ["^[\/\*\']", "[\/\*\']$"] # => ""
  def __init__(self):
    self.sub_map = {}

  # TODO: ["[a-z]/[a-z]"] => " or "
  def prune(self, tags):
    for i in xrange(len(tags)):
      for r in self.bad_affix:
        s = re.sub(r, "", tags[i])
        if s != tags[i]:
          self.sub_map[s] = tags[i]
          tags[i] = s
      for r in self.replace:
        # Don't bother to record replaced text..
        s = re.sub(r, "", tags[i])
        if s != tags[i]:
          tags[i] = s
    return tags

  # FIXME: This needs context...
  def deprune(self, pos_tags):
    for i in xrange(len(pos_tags)):
      if pos_tags[i][0] in self.sub_map:
        pos_tags[i] = (self.sub_map[pos_tags[i][0]], pos_tags[i][1])
    return pos_tags

  def agfl_repair(self, agfl_tags, nltk_tags):
    # This is somewhat hackish
    for a in xrange(len(agfl_tags)):
      if not agfl_tags[a][1] or agfl_tags[a][1] == "WORD" or \
            agfl_tags[a][1].isspace():
        closest = -1
        for n in xrange(len(nltk_tags)):
          if agfl_tags[a][0] == nltk_tags[n][0]:
            if closest > 0:
              print "Two tags found for "+agfl_tags[a][0]
              if abs(a-n) > abs(a-closest):
                continue # Farther away. Skip
            closest = n
            if nltk_tags[n][1] in self.nltk_ep4_map:
              agfl_tags[a] = (nltk_tags[n][0],
                                self.nltk_ep4_map[nltk_tags[n][1]])
            else:
              #print nltk_tags[n][1]+" not in nltk map!"
              agfl_tags[a] = (nltk_tags[n][0], nltk_tags[n][1])

agfl = AGFL.AGFLWrapper()
# XXX: Maybe rewrite tweaker.prune() to be
# position aware. Then we can stop doing quite
# so many calls to word_detokenize()
def pos_tag(tokens):
  if agfl.agfl_ok():
    detoked = word_detokenize(tokens)
    sentences = nltk.sent_tokenize(detoked)
    all_tags = []
    for s in sentences:
      stokens = word_tokenize(s)
      nltk_tags = nltk.pos_tag(stokens)
      tweaker = AGFLTweaker()
      tweaker.agfl_fix(stokens, nltk_tags)
      tweaker.prune(stokens)
      s = word_detokenize(stokens)
      if not s:
        print "Empty string for: "+str(stokens)
        continue
      print "Parsing: |"+s+"|"
      agfl_tree = agfl.parse_sentence(s)
      if not agfl_tree:
        print "Parse fail for |"+s+"|"
        return None # Hrmm. use partials? Prob not
      else:
        tags = agfl_tree.pos_tag()
        tweaker.agfl_repair(tags, nltk_tags)
        tweaker.deprune(tags)
        if tags:
          all_tags.extend(tags)
        else:
          print "Tag fail for |"+s+"|"
          return None
    return all_tags
  else:
    # XXX: Kill this log
    print "AGFL not found/functional. Fallig back to nltk.pos_tag()"
    return nltk.pos_tag(tokens)


def main():
  tweaker = AGFLTweaker()
  toks = word_tokenize("@John I am goin' to the #store, *catfeesh*?")
  tweaker.prune(toks)
  print toks
  tweaker.deprune(toks)
  print toks
  print pos_tag(word_tokenize("@John, I am going to the #store."))

if __name__ == "__main__":
   main()
