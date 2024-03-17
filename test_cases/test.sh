#!/bin/bash
> all_results.txt

for i in {1..8}; do
    input_file="case$(printf "%02d" $i).in"
    expected_output_file="case$(printf "%02d" $i).out"

    echo "Running test case $i"
    echo "Case $i:"  >> all_results.txt
    echo -n "Expected:     " >> all_results.txt
    cat "$expected_output_file" >> all_results.txt
    echo -n "Output:       " >> all_results.txt
    python3 solution.py < "$input_file" >> all_results.txt

done

echo "Finish"