The shell script wrap_pyjac takes as an input the .cti chemistry file and produces the static library .so file > pyjacob.so, to be used for pyjac.

Pyjac automatically reorders the mechanism's species and takes the first species between N2, Ar or He found in the model and places it last, as described in: 
http://slackha.github.io/pyJac/faqs.html#ordering

Examples taken from: http://slackha.github.io/pyJac/examples.html
