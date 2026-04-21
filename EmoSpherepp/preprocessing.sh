CUDA_VISIBLE_DEVICES=0 python embedding_extract.py --wav_directory "/mnt/workspace/hongchengxun/dataset/ESD" --wavlm_save_directory "/mnt/workspace/hongchengxun/dataset/ESD_emb/WavLM" --emotion2vec_save_directory "/mnt/workspace/hongchengxun/dataset/ESD_emb/Emotion2Vec"


CUDA_VISIBLE_DEVICES=0 python align_and_binarize.py --config egs/datasets/audio/esd/fs2_orig.yaml