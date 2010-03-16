Quick steps:

0. Download and install nltk-2.08b and all its dependencies.
1. (Optional, but recommended): Download AGFL and the EP4ir grammer.
2. Build the EP4ir grammar in ./libs/agfl_ep4/ with 'agfl ep4ir.gra'
3. Use the harvest.py script to pull down a bunch of .jtwt files
   from a twitter feed.
4. Throw them in the directory ./target_user/
5. Run ./resurrect.py to generate a .soul file and a .brain file.
6. Torment the AI to your hearts content.

Note that the .soul and .brain generation process takes quite some time.
The brain generation process does run in the background while you chat
with the AI as its brain forms.

