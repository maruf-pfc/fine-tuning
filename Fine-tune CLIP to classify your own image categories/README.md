# Fine-tune CLIP to classify your own image categories

## Step 1 — Installing Libraries

```python
!pip install transformers peft datasets torch torchvision --quiet
```

**What each library does:**

**transformers** — HuggingFace's library. Contains pre-trained models like CLIP, BERT, GPT. Think of it as a store where you can download any major AI model in one line of code.

**peft** — Parameter-Efficient Fine-Tuning library. Contains LoRA and other efficient fine-tuning methods. This is what her paper uses internally.

**datasets** — HuggingFace's dataset library. Easy access to thousands of public datasets including CIFAR10.

**torch** — PyTorch. The main deep learning framework. Almost all modern AI research uses this.

**torchvision** — PyTorch's computer vision extension. Image transforms, datasets, pre-trained vision models.

The `!` means run this as a terminal command inside the notebook. `--quiet` suppresses verbose output.

---

## Step 2 — Loading Pre-trained CLIP

```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Model loaded!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Line by line:**

`from transformers import CLIPModel, CLIPProcessor` — import two classes. CLIPModel is the actual neural network. CLIPProcessor handles converting raw images and text into the numerical format the model understands.

`CLIPModel.from_pretrained("openai/clip-vit-base-patch32")` — download and load the CLIP model. "openai/clip-vit-base-patch32" is the model identifier on HuggingFace Hub. This downloads ~600MB of pre-trained weights. The model already knows about millions of visual concepts from its original training on 400 million image-text pairs.

`CLIPProcessor.from_pretrained(...)` — download the processor. This handles resizing images to 224x224, normalizing pixel values, and tokenizing text into numbers the model understands.

`sum(p.numel() for p in model.parameters())` — count total parameters. `p.numel()` means "number of elements" in a parameter tensor. This loops through all parameters and sums them up. You'll see about 151 million parameters.

---

## Step 3 — Zero-Shot Classification (CLIP Before Fine-tuning)

```python
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob:.3f}")
```

**Line by line:**

`from PIL import Image` — PIL (Python Imaging Library) handles image loading and manipulation.

`requests.get(url, stream=True).raw` — downloads the image from the URL as a stream. This is a photo of two cats from the COCO dataset.

`Image.open(...)` — opens the downloaded image as a PIL Image object.

`texts = ["a photo of a cat", ...]` — these are your candidate categories written as natural language sentences. CLIP was trained on sentences like these so this phrasing works better than just "cat".

`processor(text=texts, images=image, return_tensors="pt", padding=True)` — this does several things at once. It resizes and normalizes the image, tokenizes the text into numbers, and returns PyTorch tensors ("pt"). `padding=True` pads shorter text sequences to match the longest one.

`with torch.no_grad():` — tells PyTorch not to calculate gradients during this block. Gradients are needed for training but waste memory during inference. Always use this when just making predictions.

`outputs = model(**inputs)` — the `**` unpacks the inputs dictionary into keyword arguments. The model runs a forward pass through both the image encoder and text encoder.

`outputs.logits_per_image` — a tensor of shape [1, 3]. One row (one image), three columns (three text options). Each value is a similarity score between the image and that text.

`logits.softmax(dim=1)` — converts raw similarity scores into probabilities that sum to 1.0. `dim=1` means apply softmax across the columns (across the text options).

**Your result:**
```
a photo of a cat: 0.993
a photo of a dog: 0.005
a photo of a car: 0.002
```
CLIP is 99.3% confident it's a cat. This is zero-shot — CLIP was never specifically trained to classify this image. It just learned from 400M image-text pairs that cats look a certain way.

---

## Step 4 — Adding LoRA to CLIP

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
```

**Line by line:**

`from peft import LoraConfig, get_peft_model` — import LoRA tools. LoraConfig defines the LoRA settings. get_peft_model wraps an existing model with LoRA.

`r=4` — the rank. This is the key LoRA hyperparameter. Rank 4 means the adapter matrices A and B have dimensions (d×4) and (4×d). Same rank used in her paper. Lower rank = fewer parameters but less capacity.

`lora_alpha=8` — scaling factor. The LoRA update is scaled by alpha/r = 8/4 = 2. Controls how much the LoRA update influences the original weights.

`target_modules=["q_proj", "v_proj"]` — which layers to apply LoRA to. "q_proj" is the query projection matrix in self-attention. "v_proj" is the value projection matrix. These are the most important attention components — same choice made in the original LoRA paper.

`lora_dropout=0.1` — randomly zeros 10% of LoRA activations during training. Prevents overfitting.

`bias="none"` — don't add LoRA to bias terms. Keeps things simple.

`get_peft_model(model, lora_config)` — this wraps your model. It goes through every layer named "q_proj" or "v_proj" and replaces them with LoRA versions. The original weights are frozen automatically.

`print_trainable_parameters()` — prints something like:
```
trainable params: 294,912 || all params: 151,277,056 || trainable%: 0.19%
```
Only 0.19% of parameters are being trained. Everything else is frozen. This is why LoRA is so efficient.

**Visualizing what LoRA does:**

```
Before LoRA:
Input → [Frozen W matrix] → Output

After LoRA:
Input → [Frozen W matrix] → +
Input → [Trainable A] → [Trainable B] → Output (added together)
```

W stays exactly the same. A and B are small new matrices that learn the task-specific adjustment.

---

## Step 5 — Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("cifar10", split="train[:1000]")
print(dataset)
print(dataset[0])
```

**Line by line:**

`load_dataset("cifar10", split="train[:1000]")` — downloads CIFAR10 and loads only the first 1000 training examples. CIFAR10 has 60,000 images total (50K train, 10K test) across 10 classes. We use a small subset to keep training fast on Kaggle.

`dataset[0]` — shows the first example. You'll see something like `{'img': <PIL.Image>, 'label': 6}`. The label 6 corresponds to "frog" in CIFAR10.

**CIFAR10 classes and their label numbers:**
```
0: airplane
1: automobile  
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
```

---

## Step 6 — The Fine-tuning Loop

```python
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

text_prompts = [f"a photo of a {name}" for name in class_names]

optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
lora_model = lora_model.to(device)

lora_model.train()
for epoch in range(3):
    total_loss = 0
    for i, example in enumerate(dataset):
        if i >= 100:
            break
            
        image = example["img"]
        label = example["label"]
        
        inputs = processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        outputs = lora_model(**inputs)
        logits = outputs.logits_per_image
        
        target = torch.tensor([label]).to(device)
        loss = nn.CrossEntropyLoss()(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}, Average Loss: {total_loss/100:.4f}")
```

**Line by line:**

`text_prompts = [f"a photo of a {name}" for name in class_names]` — creates 10 text descriptions, one per class. This is a list comprehension — a compact Python loop. Result: `["a photo of a airplane", "a photo of a automobile", ...]`

`torch.optim.AdamW(lora_model.parameters(), lr=1e-4)` — creates the optimizer. AdamW is the standard optimizer for transformer fine-tuning (used in LoRA paper, her paper, almost everything). `lr=1e-4` is the learning rate — how big each update step is. `lora_model.parameters()` only returns the trainable LoRA parameters since everything else is frozen.

`device = "cuda" if torch.cuda.is_available() else "cpu"` — automatically use GPU if available. On Kaggle with GPU enabled this returns "cuda". Training on GPU is ~10-50x faster than CPU.

`lora_model = lora_model.to(device)` — move all model weights to GPU memory.

`lora_model.train()` — puts model in training mode. This enables dropout (randomly zeros some activations during training for regularization). The opposite is `model.eval()` used during evaluation.

`for epoch in range(3):` — an epoch is one full pass through the training data. We do 3 epochs meaning we see each training example 3 times.

`for i, example in enumerate(dataset):` — loop through examples. `enumerate` gives both the index `i` and the example.

`if i >= 100: break` — only use 100 examples per epoch to keep it fast. For serious training you'd remove this.

`image = example["img"]` — extract the PIL image from the dataset example.

`label = example["label"]` — extract the integer class label (0-9).

`inputs = processor(...).to(device)` — process image and all 10 text prompts. The `.to(device)` moves the resulting tensors to GPU.

`outputs = lora_model(**inputs)` — forward pass. The model processes the image through the image encoder and all 10 texts through the text encoder simultaneously.

`logits = outputs.logits_per_image` — shape [1, 10]. One image, ten similarity scores. The score at position `i` is how similar the image is to text prompt `i`.

`target = torch.tensor([label]).to(device)` — wrap the integer label in a tensor and move to GPU. This is the correct answer index.

`loss = nn.CrossEntropyLoss()(logits, target)` — CrossEntropy loss measures how wrong the prediction is. It wants the logit at position `label` to be higher than all other logits. If the model correctly assigns highest similarity to the correct class, loss is low. If it assigns highest to the wrong class, loss is high.

**Simple example:** If the image is a cat (label=3) and logits are:
```
[0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
```
Position 3 (cat) has highest score → low loss. Good prediction.

```
[0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
```
Position 5 (dog) has highest score → high loss. Wrong prediction.

`optimizer.zero_grad()` — clear gradients from the previous step. PyTorch accumulates gradients by default so you must clear them manually each step.

`loss.backward()` — backpropagation. Calculates how much each trainable parameter (only the LoRA A and B matrices) contributed to the loss. Stores these as gradients.

`optimizer.step()` — update the LoRA parameters using the gradients. Moves them slightly in the direction that reduces loss.

`total_loss += loss.item()` — `.item()` extracts the scalar value from a tensor. Accumulate for averaging.

**Training loop summary in plain English:**
1. Show the model an image
2. Ask "which of these 10 descriptions matches this image?"
3. Check if it got the right answer
4. Calculate how wrong it was (loss)
5. Figure out which LoRA parameters caused the mistake (backward)
6. Nudge those parameters to do better next time (step)
7. Repeat 100 times per epoch, 3 epochs

---

## Step 7 — Evaluation

```python
lora_model.eval()
correct = 0
total = 0

test_dataset = load_dataset("cifar10", split="test[:200]")

with torch.no_grad():
    for example in test_dataset:
        image = example["img"]
        label = example["label"]
        
        inputs = processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        outputs = lora_model(**inputs)
        predicted = outputs.logits_per_image.argmax().item()
        
        if predicted == label:
            correct += 1
        total += 1

print(f"Accuracy after fine-tuning: {correct/total*100:.1f}%")
```

**Line by line:**

`lora_model.eval()` — switch to evaluation mode. Disables dropout so predictions are deterministic.

`test_dataset = load_dataset("cifar10", split="test[:200]")` — load 200 test examples. These images were never seen during training.

`with torch.no_grad():` — disable gradient computation. Not training here, just predicting. Saves memory and speeds things up.

`logits_per_image.argmax()` — `.argmax()` returns the index of the highest value. If position 3 has the highest similarity score, argmax returns 3, meaning the model predicts "cat".

`.item()` — converts single-element tensor to a Python integer.

`if predicted == label:` — check if the predicted class matches the true class.

**Your result: 93.0% accuracy**

200 test images, 186 correct predictions. This is strong for such simple fine-tuning.

---

## What The Numbers Mean

**Before fine-tuning (zero-shot CLIP):** Already works well for categories like "cat" because CLIP was trained on internet data containing cats. Zero-shot accuracy on CIFAR10 with standard CLIP is around 65-75%.

**After fine-tuning with LoRA (your result): 93.0%** — significant improvement. With only 100 examples per epoch × 3 epochs = 300 total training steps and only 0.19% of parameters updated.

**What this demonstrates about LoRA:** Even with rank r=4 (very small adapters), the model learns task-specific improvements efficiently. The original CLIP knowledge is preserved (the frozen weights) while the LoRA adapters add task-specific knowledge on top.
