# EEG-Based Emotion Recognition with Transformers

Code and experiments for a diploma thesis.

This work evaluates five deep learning architectures on emotion recognition from EEG
signals and critically examines the high accuracies commonly reported in the literature.
The central finding is that much of that performance stems from **subject identity
leakage**: when windows from the same participant appear in both training and test sets,
the networks learn to recognise *who* produced the signal rather than *what* they felt.

---

## Evaluation protocol

Every model is evaluated at three levels of strictness:

| Level | Split | What it measures |
|---|---|---|
| **S1** | Random, window-level | The optimistic setting used in much of the literature |
| **S2** | GroupKFold by trial | Intermediate — no temporal overlap |
| **S3** | Leave-One-Subject-Out | Realistic — an unseen user |

The gap between S1 and S3 quantifies the leakage.

## Results

Balanced Accuracy / Macro F1 (%), binary valence classification.

| Model | Dataset | S1 | S2 | S3 | Majority | Drop (pp) |
|---|---|---|---|---|---|---|
| ACTNN | DREAMER | 64.56 / 64.86 | 49.99 / 48.99 | 46.07 / 43.56 | 61.11 | **18.49** |
| ACTNN | SEED-IV | 60.09 / 60.07 | 44.90 / 44.72 | 50.78 / 48.25 | 50.47 | 9.31 |
| CCNN | DREAMER | 53.03 / 44.58 | 49.44 / 40.23 | 48.98 / 40.60 | 60.89 | 4.05 |
| CCNN | DEAP | 62.57 / 64.16 | 54.16 / 53.94 | 48.29 / 47.01 | 77.73 | 14.27 |
| CNN-LSTM | DREAMER | 71.31 / 71.91 | 49.86 / 45.34 | 50.56 / 45.60 | 61.11 | **20.74** |
| CNN-LSTM | DEAP | 61.78 / 63.43 | 56.22 / 56.53 | 50.01 / 40.23 | 77.73 | 11.77 |
| EEGDINO | DREAMER | 51.32 / 41.35 | 49.87 / 39.51 | 49.26 / 38.96 | 61.11 | 2.05 |
| EEGDINO | DEAP | 56.34 / 56.09 | 54.10 / 53.29 | 49.06 / 43.23 | 77.73 | 7.29 |
| MozartussNet | DEAP | 61.84 / 64.02 | 52.49 / 49.89 | 50.01 / 40.86 | 77.73 | 11.84 |

**Almost no architecture beats the majority-class baseline under S3.** A small drop is not
good news either: EEGDINO loses only 2.05 points because it fails equally at all three
levels, so there was no performance left to lose.

## Representation analysis

Beyond accuracy, three complementary analyses were applied:

- **Linear CKA** between the S1 and S3 models at matching stages. Early layers produce
  near-identical representations (0.73–1.00); final layers diverge sharply (0.04–0.70).
  Feature extraction is consistent — the failure appears exactly where the network decides
  what to keep.
- **Layer-wise probing** with linear classifiers on frozen representations. The emotion
  label stays at 0.94–1.40× the majority baseline, while subject identity reaches
  **27.24×** chance (CCNN/DEAP, 99.89% accuracy) and trial identity 7.69×.
- **Separability measures** per stage: Silhouette, Davies-Bouldin, Fisher ratio, and the
  inter/intra class distance ratio.

Where enough subjects were available, the drop was statistically significant: Wilcoxon
p = 0.0013 (CCNN/DREAMER) and p = 0.0007 (CCNN/DEAP).

## Explainability (XAI)

Up to ten methods per model: Integrated Gradients, DeepLIFT, Gradient SHAP, KernelSHAP,
LIME, Saliency, Input×Gradient, Grad-CAM, Occlusion Sensitivity, Feature Ablation and
Permutation Importance.

Every explanation is validated quantitatively with:
- **AOPC deletion/insertion** against random deletion (faithfulness)
- **Cascading parameter randomisation** (sanity check)
- **Spearman and Jaccard\@5** between methods (agreement)

Two findings stand out. Faithfulness scores collapse wherever the model itself collapsed
to single-class prediction. Conversely, good faithfulness guarantees nothing: CNN-LSTM had
the worst generalisation and the highest AOPC scores of all.

---

## Layout

```
notebooks/   One notebook per model-dataset pair.
             Saved outputs (tables and figures) are included.
```

## Datasets

None are included in this repository — all require an access request.

| Dataset | Subjects | Channels | Access |
|---|---|---|---|
| DEAP | 32 | 32 | https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ |
| DREAMER | 23 | 14 | https://zenodo.org/record/546113 |
| SEED-IV | 15 | 62 | https://bcmi.sjtu.edu.cn/home/seed/ |

## Running

The notebooks were written for **Kaggle with a GPU** (T4 or P100). To run locally:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Dataset paths are set in the first cells of each notebook and need to be adjusted.

## Libraries

PyTorch · TensorFlow/Keras · TorchEEG · Braindecode · Captum · SHAP · LIME · MNE ·
scikit-learn · SciPy · pandas · matplotlib · seaborn

## License

MIT — see `LICENSE`.
