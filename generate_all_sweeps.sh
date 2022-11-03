#!/bin/bash

all_nets=(all)
#var=$(date '+%d-%m-%Y')
echo Hello, what would you like to call this project?
read projname
rm sweeps.txt
for n in ${all_nets[@]}; do
  echo $n
  sweep_str=$n.yaml
  cp "$sweep_str" "$sweep_str.temp"
  echo "project: Sweep_$projname" > "$sweep_str"
  cat "$sweep_str.temp" |tail -n+2>> "$sweep_str"
  wandb sweep $sweep_str >.temp 2>&1
  tail -1 .temp > .temp2 2>&1
  line=$(cat .temp2)
  echo ${line:29:54} >> sweeps.txt
rm "$sweep_str.temp"

done
rm .temp
rm .temp2