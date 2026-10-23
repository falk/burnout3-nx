#!/bin/bash
# fps.sh log: frames per second and milliseconds per frame over 10 s windows, from the null renderer's
# "renderer: frame N, D draws" lines (an nxlink capture or logs/burnout3_001.log); draws/frame above ~500 is in a level
sed -nE 's/.*\[ *([0-9.]+)\].*renderer: frame ([0-9]+), ([0-9]+) draws.*/\1 \2 \3/p' "${1:?usage: fps.sh log}" |
    awk 'NR == 1 { pt = $1; pf = $2; pd = $3; next }
         $1 - pt >= 10 && $2 > pf { fps = ($2 - pf) / ($1 - pt)
             printf "t=%6.1fs  %6.1f fps  %6.2f ms/frame  %5.0f draws/frame\n", $1, fps, 1000 / fps, ($3 - pd) / ($2 - pf)
             pt = $1; pf = $2; pd = $3 }'
