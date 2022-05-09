#!/bin/bash

all_nets=(custom)

for n in ${all_nets[@]}; do
  sweep_str=$n.yaml
  wandb sweep $sweep_str >.temp 2>&1
  tail -1 .temp > .temp2 2>&1
  line=$(cat .temp2)
  echo ${line:29:42} >> sweeps.txt


done
rm .temp
rm .temp2