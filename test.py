import json
import tensorflow as tf

model = tf.keras.models.load_model("model/cats_dogs_transfer.keras")

with open("model/class_names.json", encoding="UTF-8") as fhand:
    class_names = json.load(fhand)

print("Input shape:", model.input_shape)
print("output shape:", model.output_shape)
print("class names:", class_names)

assert model.input_shape[1:] == (160, 160, 3), "Unexpected input size"
assert model.output_shape[-1] == len(class_names), "Label count mismatch"
print("Modle and class mapping are consistent.")