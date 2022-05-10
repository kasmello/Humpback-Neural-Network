#!/bin/bash

input="./sweeps.txt"
while IFS= read -r line
do
  echo $line
  $line
done < $input