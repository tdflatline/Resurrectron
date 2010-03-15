#!/usr/bin/python

import time
import subprocess
#from AsyncPopen import Popen
from subprocess import Popen

class EP4irParseTree:
  upenn_map = { "NOUN(sing)":"NN" }
  def __init__(self, parseresult):
    parseresult = parseresult.replace("epos{}", "")
    parseresult = parseresult.replace("spos{}", "")
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
      pos_end = l.find('"')
      tag = l[:pos_end]
      word = l[pos_end+1:l.find('"', pos_end+1)]

      if not word:
        return None # XXX: this happens.

      if not tag and word[0] == "-":
        # punctuation
        tag = word
        word = word[1:]
      tags.append((word, tag))
    return tags

  def pos_tag_upenn(self):
    pass

class AGFLWrapper:
  # XXX: Better errro handling for pipe failure
  def __init__(self):
    # agfl-run -T 1 -B -b ep4ir
    self.p = Popen("agfl-run -v totals -H -T 1 -B -b ./libs/agfl_ep4/ep4ir", shell=True,
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
  tree = ep.parse_sentence("I love you.")
  print "T: |"+str(tree.tree)+"|"
  print "L: |"+str(tree.leaves)+"|"
  print str(tree.pos_tag())

if __name__ == "__main__":
  main()
