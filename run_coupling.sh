#!/bin/bash
echo "Starting evaluation of coupling coefficients..."

python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 0,1,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 0,1,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 1,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 1,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 0,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 0,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 2,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 2,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 1,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 1,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 1,1,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 1,1,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 0,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 0,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 1,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 1,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 2,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 2,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 1,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 1,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 1,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 1,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 2,1,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 2,1,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 0,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 0,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 2,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 2,1,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 0,2,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 0,2,0 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 1,2,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 1,2,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 2,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 2,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 0,2,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 0,2,1 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 2,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 2,1,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 1,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 1,1,3 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 1,2,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 1,2,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol cross --mode-fam TM --mode-ind 0,2,2 --N-theta 50 --nproc 8
python3 coupling_strength.py --pol plus --mode-fam TM --mode-ind 0,2,2 --N-theta 50 --nproc 8

#dead ones. TO DO: WHY?????? integraton dying
# python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 0,1,2 --N-theta 50 --nproc 8
# python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 0,1,2 --N-theta 50 --nproc 8
# python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 0,1,3 --N-theta 50 --nproc 8
# python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 0,1,3 --N-theta 50 --nproc 8
# python3 coupling_strength.py --pol cross --mode-fam TE --mode-ind 0,1,1 --N-theta 50 --nproc 8
#python3 coupling_strength.py --pol plus --mode-fam TE --mode-ind 0,1,1 --N-theta 50 --nproc 8