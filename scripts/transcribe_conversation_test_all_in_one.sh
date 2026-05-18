#!/usr/bin/env bash
input=$1

./transcribe_conversation_by_paraformer.sh "$input"
./transcribe_conversation_by_nano.sh "$input"
./transcribe_conversation_by_sensevoice.sh "$input"