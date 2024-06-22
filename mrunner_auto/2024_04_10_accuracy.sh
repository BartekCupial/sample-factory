#!/bin/bash

ssh-add

mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/accuracy/2024_04_10_monk-APPO-KS-T-accuracy.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/accuracy/2024_04_10_monk-APPO-KS-T-accuracy.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/accuracy/2024_04_10_monk-APPO-BC-T-accuracy.py
mrunner --config ~/.mrunner.yaml --context athena_nethack_big_1gpu run mrunner_exps/accuracy/2024_04_10_monk-APPO-T-accuracy.py
