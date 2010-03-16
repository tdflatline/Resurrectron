Quick steps:

0. Download and install nltk-2.08b and all its dependencies:
   http://www.nltk.org/
1. Optional, but recommended: Download AGFL 2.8:
   http://www.agfl.cs.ru.nl/download.html
   http://www.cs.ru.nl/cdl3/
2. Optional: Build the EP4ir grammar in ./libs/agfl_ep4/ with 'agfl ep4ir.gra'
3. Use the harvest.py script to pull down a bunch of .jtwt files
   from a twitter feed.
4. Throw them in the directory ./target_user/
5. Run ./resurrect.py to generate a .soul file and a .brain file.
6. Torment the AI to your hearts content.

Note that the .soul and .brain generation process takes quite some time.
The brain generation process does run in the background while you chat
with the AI as its brain forms.

This is still hackware. Don't expect it to be too useful if you don't
know python.
