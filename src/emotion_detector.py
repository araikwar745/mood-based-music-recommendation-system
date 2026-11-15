import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionDetector:
    """Loads a GoEmotions model and predicts one or more emotion labels."""

    def __init__(self, model_path, tokenizer_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The tokenizer MUST match the model architecture. The public
        # SamLowe model is RoBERTa, not DistilBERT.
        tokenizer_name = tokenizer_name or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).int().cpu().numpy()
        predicted_indices = predictions[0].nonzero()[0]
        if len(predicted_indices) > 0:
            return [self.model.config.id2label[int(i)] for i in predicted_indices]
        return ["neutral"]
