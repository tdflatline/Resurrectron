Quick steps:

0. Download and install nltk-2.08b and all its dependencies:
   http://www.nltk.org/
1. Download and install the NodeBox linguistics library:
   http://nodebox.net/code/index.php/Linguistics (we use 1.9.4.2).
   This one just needs to be in your $PYTHONPATH
2. Optional, but strongly recommended: Download AGFL 2.8 and CDL3:
   http://www.agfl.cs.ru.nl/download.html
   http://www.cs.ru.nl/cdl3/
   You may need to apply the patch in ./patches/cdl-1.2.7.diff to get
   CDL3 to compile.
3. Optional: Build the EP4ir grammar in ./libs/agfl_ep4/ with 'agfl ep4ir.gra'
4. Use the harvest_*.py scripts to pull down a bunch of .jtwt files
   from a twitter feed, or posts from a blog. You can also create .twt files
   by hand, which can contain 1 quote/quip per line.
5. Throw them in the directory ./target_user/
6. Optional: Tweak settings in ./settings.cfg for your use case.
7. Run ./resurrect.py to generate a .soul file and a .brain file.
8. Torment the AI to your hearts content.
9. Hit Control-D, Control-C, or :q!<enter> to exit.

Note that the .soul and .brain generation process takes quite some time.
The brain generation process does run in the background while you chat
with the AI as its brain forms, but some of the answers may be delayed
during this process.

This is still hackware. Don't expect it to be too useful if you don't
know python.
