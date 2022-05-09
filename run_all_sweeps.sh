#!/bin/bash

input="./sweeps.txt"
while IFS= read -r line
do
  $line
done 