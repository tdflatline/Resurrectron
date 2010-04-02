#!/usr/bin/python

import time
import subprocess
#from AsyncPopen import Popen
from subprocess import Popen
import re

class EP4irParseTree:
  upenn_map = { "NOUN(sing)":"NN" }
  def __init__(self, parseresult):
    parseresult = parseresult.replace("epos{}", "")
    parseresult = parseresult.replace("spos{}", "")
    parseresult = parseresult.replace("PRICE{}", "")
    self.tree = self._make_tree(parseresult)
    self.leaves = self._gather_leaves(self.tree)

  def _find_end_brace(self, string):
    # "{} hi {}" and "root{hi{} and{} hi{}}"
    open_count = 0
    for i in xrange(len(string)):
      if string[i] == "{":
        open_count += 1
      if string[i] == "}":
        open_count -= 1
        if open_count == 0:
          return i

  def _make_tree(self, string):
    # Base case:
    if not string: return []

    begin_subtree = string.find("{")
    end_subtree = self._find_end_brace(string)

    if begin_subtree == -1:
      return [string]
    elif end_subtree == -1:
      print "WTF??: "+string
      return [string]
    else:
      #print "P: "+str(string[begin_subtree+1:end_subtree])
      #print "R: "+str(string[end_subtree+1:])
      subtree = self._make_tree(string[begin_subtree+1:end_subtree])
      if subtree:
        retlist = [string[0:begin_subtree], subtree]
      else:
        retlist = [string[0:begin_subtree]]
      retlist.extend(self._make_tree(string[end_subtree+1:]))
      return retlist

  def _gather_leaves(self, tree):
    if len(tree) == 1: return [tree[0]]
    leaves = []

    for br in xrange(len(tree)):
       if type(tree[br]) == list:
         leaves.extend(self._gather_leaves(tree[br]))
    return leaves

  def pos_tag(self):
    tags = []
    for l in self.leaves:
      # TODO: This is a hack. Wtf is this IT{} leaf?
      if l == "IT": continue

      pos_end = l.find('"')
      tag = l[:pos_end]
      word = l[pos_end+1:l.find('"', pos_end+1)]

      if not word:
        print "Word blank for leaf: "+str(l)
        # XXX: Some words have weird tags...
        # leave them for nltk.pos_tag()
        # XXX: This may be wrong..
        tags.append((tag, ""))
        continue

      # FIXME: Sometimes AGFL cites a parent leaf with [1] or [2]..
      # Usually it does this when its broken, but sometimes it actually
      # has correctly inferred a NOUN/VERB POS position.

      # AGFL sucks at labeling #'s
      if word.isdigit():
        tag = "NUM(card)"

      # it also sucks at "I".
      if not tag and (word == "I" or word == "i"):
        tag = "PERSPRON(sing,first,nom)"

      if not tag and word[0] == "-":
        # punctuation
        tag = word
        if word[-1] == "-":
          word = word[1:-1]
        else:
          word = word[1:]

      # XXX: Also ADJET(TEXT), ADVBT(TEXT,ADVT), PREPOST(TEXT,PREP),
      # PAIRDET(TEXT)?
      if tag.startswith("VERB"):
        tag = re.sub(r"^(VERB[A-Z]\()[^,]+", r"\1NONE", tag)

      tags.append((word, tag))
    return tags

  def pos_tag_upenn(self):
    pass

class AGFLWrapper:
  def __init__(self):
    # agfl-run -T 1 -B -b ep4ir
    self.p = Popen("agfl-run -v totals -H -T 2 -B -b ./libs/agfl_ep4/ep4ir", shell=True,
              bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
              stderr=subprocess.STDOUT, close_fds=True)

  def agfl_ok(self):
    return self.p.poll() == None

  # AGFL only correctly handles a sentence at a time.
  def parse_sentence(self, sentence, ParseTree=EP4irParseTree):
    if not sentence.endswith("\n"): sentence += "\n"
    if self.p.poll() != None: return None
    self.p.stdin.write(sentence)
    result = ""
    result_part = ""
    while not result_part or "# parsings" not in result:
      if self.p.poll() != None: return None
      result_part = self.p.stdout.readline()
      if result_part: result += result_part
    ret = result.split("\n")
    #print ret
    if ret[0].startswith("# parsings 0"):
      return None
    return ParseTree(ret[0])


def main():
  ep4 = EP4irParseTree("line{hi{hi}}line{epos{} bye spos{}}")
  ep = AGFLWrapper()
  tree = ep.parse_sentence("Am I going to get drunk tonight?")
  print "T: |"+str(tree.tree)+"|"
  print "L: |"+str(tree.leaves)+"|"
  print str(tree.pos_tag())

if __name__ == "__main__":
  main()
