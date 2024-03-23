#!/bin/bash
datasets=("alien" "amidar" "assault" "asterix" "asteroid" "atlantis" "bankheist" "battlezone" "beamrider" "berzerk" "bowling" "boxing" "breakout" "centipede" "choppercommand" "crazyclimber" "defender" "demonattack" "doubledunk" "enduro" "fishingderby" "freeway" "frostbite" "gopher" "gravitar" "hero" "icehockey" "jamesbond" "kangaroo" "krull" "kongfumaster" "montezuma" "mspacman" "namethisgame" "phoenix" "pitfall" "pong" "privateye" "qbert" "riverraid" "roadrunner" "robotank" "seaquest" "skiing" "solaris" "spaceinvaders" "stargunner" "surround" "tennis" "timepilot" "tutankham" "upndown" "venture" "videopinball" "wizardofwor" "yarsrevenge" "zaxxo")

for envname in ${datasets[@]}; do
 	python -m sample_factory.huggingface.load_from_hub -r edbeeching/atari_2B_atari_${envname}_1111 -d ./train_dir
	python -m sf_examples.atari.enjoy_atari --env=atari_${envname} --experiment=atari_2B_atari_${envname}_1111  --train_dir=./train_dir/ --no_render --save_video --max_num_frames 100_000 --use_record_episode_statistics True
done
