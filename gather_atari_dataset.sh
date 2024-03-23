envname=$1

python -m sample_factory.huggingface.load_from_hub -r edbeeching/atari_2B_atari_${envname}_1111 -d ./train_dir
python -m sf_examples.atari.enjoy_atari --env=atari_${envname} --experiment=atari_2B_atari_${envname}_1111  --train_dir=./train_dir/ --no_render --save_video --max_num_frames 100_000
