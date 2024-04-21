#!/bin/bash

# Define paths
MODEL_PATH="data/POS-tagging/fr-train.model"
OUTPUT_PATH="data/POS-tagging"
TEST_FILES=("UD_French-FQB/fr_fqb-ud-test.word"
            "UD_French-GSD/fr_gsd-ud-test.word"
            "UD_French-ParTUT/fr_partut-ud-test.word"
            "UD_French-PUD/fr_pud-ud-test.word"
            "UD_French-Sequoia/fr_sequoia-ud-test.word"
            "UD_French-Spoken/fr_spoken-ud-test.word"
            "fr-test.word")

# Create the header for the CSV file
HEADER="Test Corpus;Total Errors;Total Words;Error Rate"
for STATE in ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X; do
  HEADER="$HEADER;$STATE"
done
echo "$HEADER" > error_summary.csv

# Process each test file
for TEST_FILE in "${TEST_FILES[@]}"; do
    NAME=$(basename "$TEST_FILE" ".word")

    # Run the HMM test and error comparison
    ./use_hmm test "$MODEL_PATH" "$OUTPUT_PATH/$TEST_FILE" "$OUTPUT_PATH/$NAME.output"
    ERROR_OUTPUT=$(python3 compErrors.py "$OUTPUT_PATH/$NAME.output" "$OUTPUT_PATH/${TEST_FILE%.word}.word-pos")

    # Parse the error output
    TOTAL_ERRORS=$(echo "$ERROR_OUTPUT" | grep 'Errors:' | awk -F'/' '{print $1}' | awk '{print $2}')
    TOTAL_WORDS=$(echo "$ERROR_OUTPUT" | grep 'Errors:' | awk -F'/' '{print $2}' | awk '{print $1}')
    ERROR_RATE=$(echo "$ERROR_OUTPUT" | grep 'Errors:' | awk '{print $3}' | tr -d '()')
    # Initialize CSV line with test corpus name and error rate
    CSV_LINE="$NAME;$TOTAL_ERRORS;$TOTAL_WORDS;$ERROR_RATE"

    # Add error rates for each state to the CSV line
    for STATE in ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X; do
        if [ "$STATE" == "X" ]; then
            STATE_ERROR_RATE=$(echo "$ERROR_OUTPUT" | grep -w "$STATE" | awk '{print $4}' | tr -d '()%')
        else
            STATE_ERROR_RATE=$(echo "$ERROR_OUTPUT" | grep -w "$STATE" | awk '{print $4}' | tr -d '()%')
        fi
        CSV_LINE="$CSV_LINE;$STATE_ERROR_RATE"
    done


    # Write the CSV line to the summary file
    echo "$CSV_LINE" >> error_summary.csv
done

# Print the summary file
column -s, -t < error_summary.csv
