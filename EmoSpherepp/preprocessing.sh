set -e

# Ensure NLTK resources are available for g2p_en on newer nltk versions.
python - <<'PY'
import nltk
for pkg in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY

CUDA_VISIBLE_DEVICES=0 python embedding_extract.py --wav_directory "/mnt/workspace/hongchengxun/dataset/ESD" --wavlm_save_directory "/mnt/workspace/hongchengxun/dataset/ESD_emb/WavLM" --emotion2vec_save_directory "/mnt/workspace/hongchengxun/dataset/ESD_emb/Emotion2Vec"


CUDA_VISIBLE_DEVICES=0 python align_and_binarize.py --config egs/datasets/audio/esd/fs2_orig.yaml
