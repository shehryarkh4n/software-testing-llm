import sacrebleu

def compute_bleu(predictions, references, smoothing=1, **kwargs):
    bleu = sacrebleu.corpus_bleu(predictions, [references], smooth_method="exp" if smoothing else "none")
    return bleu.score