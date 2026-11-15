MODEL LOADING FIX FOR STREAMLIT CLOUD

Why the old app failed:
app.py pointed to ./fine-tuned-goemotions-model, but that trained model folder
is not present in the two ZIP files or in the GitHub repository. Transformers
then tried to interpret that invalid local path as a Hugging Face repository ID.

This patch uses the already fine-tuned public GoEmotions model:
https://huggingface.co/SamLowe/roberta-base-go_emotions
The matching tokenizer is loaded from the same model repository (RoBERTa).
The app no longer expects a local fine-tuned-goemotions-model folder.

HOW TO APPLY:
1. Extract this ZIP in the root of your cloned project.
2. Allow it to replace app.py.
3. Copy src/emotion_detector.py into your existing src folder, replacing that file.
4. Keep your Spotify CSV at data/data.csv. The GoEmotions dataset ZIP is not
   required for inference when using this hosted, already fine-tuned model.
5. Commit and push:
   git add app.py src/emotion_detector.py
   git commit -m "Fix GoEmotions model loading for Streamlit Cloud"
   git push origin main
6. In Streamlit Cloud, open Manage app > Reboot app.

IMPORTANT:
- First startup downloads the model from Hugging Face and may take several minutes.
- This model is approximately 0.5 GB. Streamlit Cloud must have enough memory.
- The 1,000,000 synthetic ratings in the old app have been reduced to 50,000
  because they are randomly generated placeholders, not actual user listening data.
- The original GoEmotions ZIP contains training/evaluation data, not a trained
  Transformers checkpoint (config.json + model weights).
