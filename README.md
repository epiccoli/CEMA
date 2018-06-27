# CEMA

## 1) Install Cantera and Pyjac 

- Cantera: https://www.cantera.org/docs/sphinx/html/install.html
- Pyjac: https://github.com/SLACKHA/pyJac

## 2) Test Pyjac

Test codes to see if Pyjac is successfully installed and usable with Cantera ([test folder](./tests))

1. first go into the folder "0_pyjac_wrap_example" and run wrap_pyjac.sh in the folder "Li_2003" (it uses the hydrogen scheme by Li et al. 2003)
2. if running the sh script goes well, run the script "test_pyjacob.py" in the folder "1_example_use_pyjacob"

if both scripts work without errors, this means that pyjac is fully operational.


