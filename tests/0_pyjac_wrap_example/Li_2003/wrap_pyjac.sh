#!/bin/bash

# given a .cti chemistry file, produces a pyjacob static library file that can be imported in python 

python -m pyjac --lang c --input Li_2003.cti 

python -m pyjac.pywrap --source_dir ./out/ --lang c 
  
