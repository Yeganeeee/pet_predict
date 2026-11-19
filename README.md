# Cats vs Dogs Classifier

A small web application that classifies an uploaded photograph as a cat or a dog.

## Project status

- [x] Session 1 — repository, model, inference script
- [x] Session 2 — Streamlit web interface
- [ ] Session 3 — polish, documentation, final testing

## Model

- Architecture: MobileNetV2 (ImageNet weights) + new 2-class classification head
- Input size: 160 × 160 × 3 (RGB)
- Training data: Cats and Dogs filtered dataset (2,000 train / 500 validation / 500 test)
- Saved as: `model/cats_dogs_transfer.keras`
- Class order: stored in `model/class_names.json`

## How to run the inference script
pip install -r requirements.txt
python predict.py samples/cat_sample.jpg

## Author

Yegane