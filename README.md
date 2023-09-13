# Reinforcement-Learning

conda env export | grep -v "^prefix: " > environment.yml

or

conda env export | findstr -v "^prefix: " > environment.yml

conda create -n envname python=3.10

conda env create -f environment.yml