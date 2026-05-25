#!/bin/bash
# python3 slice_integration.py --R 0.2295 --L 0.5 --Ns 250 --geometry cylindrical --Bz 3 --nproc 8 --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,0
# python3 slice_integration.py --R 0.2295 --L 0.5 --Ns 250 --geometry cylindrical --Bz 3 --nproc 8 --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,1
# python3 slice_integration.py --R 0.2295 --L 0.5 --Ns 250 --geometry cylindrical --Bz 3 --nproc 8 --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,2

# python3 rhs.py --Ns 250 --Nt 10000 --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,0 --pre-RHS --data GW_BIRME
# python3 rhs.py --Ns 250 --Nt 10000 --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,1 --pre-RHS --data GW_BIRME
# python3 rhs.py --Ns 250 --Nt 10000 --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,2 --pre-RHS --data GW_BIRME

python3 ode.py --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,0 --extend 5 --data GW_BIRME --Q 2.19e5 --onset-smoothing --Ns 250 
python3 ode.py --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,1 --extend 5 --data GW_BIRME --Q 1.81e5 --onset-smoothing --Ns 250 
python3 ode.py --theta 45.0 --phi 0.0 --mode-fam TM --mode-par b --mode-ind 0,1,2 --extend 5 --data GW_BIRME --Q 2.09e5 --onset-smoothing --Ns 250 