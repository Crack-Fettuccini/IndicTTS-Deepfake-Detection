# IndicTTS Deepfake Detection

This repository contains a deepfake detection pipeline for the IndicTTS Deepfake Challenge dataset. The pipeline uses a combination of Mel Spectrogram features, Wav2Vec2 embeddings, and ensemble learning with LightGBM and XGBoost to detect synthetic speech.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Ensemble Learning](#ensemble-learning)
- [Inference & Submission](#inference--submission)
- [Results](#results)

## Installation
To install the required dependencies, run:

```bash
pip install lightgbm xgboost numpy datasets librosa pandas torchaudio torch scikit-learn torchvision tqdm evaluate transformers accelerate bitsandbytes peft trl wandb
```

## Dataset
The dataset is loaded from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data")
train_data, test_data = dataset["train"], dataset["test"]
```

## Preprocessing
1. Convert audio to Mel Spectrogram features.
2. Extract Wav2Vec2 embeddings using `facebook/wav2vec2-base-960h`.
3. Standardize features and perform stratified splitting.

## Model Training
- Uses `Wav2Vec2ForSequenceClassification` for classification.
- Freezes feature extraction layers to optimize training.
- Fine-tunes Wav2Vec2 model using `Trainer` from Hugging Face.

```python
trainer = Trainer(
    model=wav2vec_model,
    args=training_args,
    train_dataset=filtered_train_data,
    eval_dataset=filtered_valid_data,
    compute_metrics=compute_metrics
)
trainer.train()
```

## Evaluation
Performance metrics used:
- Accuracy
- F1-score
- ROC-AUC

```python
eval_results = trainer.evaluate()
print(f"Evaluation Metrics: {eval_results}")
```

## Ensemble Learning
An ensemble of three models is used:
1. Wav2Vec2-based classifier
2. LightGBM model on Mel Spectrogram features
3. XGBoost model on Mel Spectrogram features

Final predictions are obtained using a meta LightGBM model.

```python
meta_model = lgb.train({"objective": "binary", "metric": "auc", "learning_rate": 0.05}, lgb_train, num_boost_round=100)
meta_test_preds = meta_model.predict(stacked_test_scaled)
```

## Inference & Submission
The final predictions are saved as a CSV file:

```python
submission = pd.DataFrame({"id": [sample["id"] for sample in test_data], "is_tts": meta_test_preds})
submission.to_csv("submission.csv", index=False)
```

## Results
The final submission file `submission.csv` contains the model's predictions on the test dataset.

