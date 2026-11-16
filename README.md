# 🌿 Plant Disease Recognition System

A deep-learning web app that identifies the disease of a plant from a **photo of a single leaf**.
Upload a leaf image and the app tells you the crop, whether the leaf is healthy or diseased,
how sure the model is, and the three most likely possibilities.

Built with a **Convolutional Neural Network (CNN)** in TensorFlow/Keras and a **Streamlit** web interface.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

---

## ✨ Features

- Recognises **38 classes** – 26 diseases and 12 healthy conditions on **14 crops**
- Simple web interface: upload → **Predict** → result
- Shows the **confidence %** and a **Top-3** table
- Warns when the model is not sure (confidence below 50%)
- Clear error messages instead of crashes (missing model, invalid image)
- Training script, model-check script and Jupyter notebooks included

**Crops covered:** apple, blueberry, cherry, corn (maize), grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, tomato.

## 🖼️ Screenshots

| Home | Prediction |
|---|---|
| ![Home](screenshots/home.png) | ![Prediction](screenshots/prediction.png) |

> Put your own screenshots in a `screenshots/` folder with these names.

## 🧠 How it works

1. The uploaded image is converted to RGB and resized to **128 × 128**.
2. It passes through a CNN: **5 blocks** of (Conv → Conv → MaxPool) with 32, 64, 128, 256, 512 filters,
   then Dropout (0.25), Dense 1500 (ReLU), Dropout (0.4) and a **Dense 38 softmax** output.
3. The 38 output probabilities are sorted; the highest is the prediction and the top 3 are shown.

| Item | Value |
|---|---|
| Dataset | [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) – PlantVillage based |
| Images | 70,295 train · 17,572 validation |
| Input size | 128 × 128 × 3 (raw pixel values 0–255) |
| Parameters | 7,842,762 |
| Optimizer / loss | Adam (lr 0.0001) / categorical cross-entropy |
| Epochs / batch | 10 / 32 |
| Validation accuracy | about 94% |

## 📁 Project structure

```
plant_disease_recognition/
├── main.py                       # Streamlit web app
├── plant_labels.py               # class names, model loading, preprocessing, prediction
├── train.py                      # train the model from the command line
├── check_model.py                # check that the model file is really trained
├── Train_plant_disease.ipynb     # training notebook
├── Test_plant_disease.ipynb      # single-image testing notebook
├── requirements.txt
├── home_page.jpeg                # banner for the Home page
├── sample_images/                # sample leaf photos
└── trained_plant_disease_model.keras   # trained model (create it with train.py)
```

## 🚀 Getting started

### 1. Clone and open the folder

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment and install packages (Python 3.11)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
python -m pip install -r requirements.txt
```

### 3. Get the trained model

The app needs the file `trained_plant_disease_model.keras` next to `main.py`.
Either use the model file from this repository (if included), or train your own:

```bash
python train.py "path/to/dataset"          # folder that contains train/ and valid/
python train.py "path/to/dataset" 10       # optional: number of epochs
```

Training takes about 4–5 minutes per epoch on a GPU and much longer on a CPU.

Check that the model is trained:

```bash
python check_model.py "path/to/dataset"
```

### 4. Run the app

```bash
python -m streamlit run main.py
```

Open **http://localhost:8501**, choose **Disease Recognition**, upload a leaf image and press **Predict**.

## 📂 Dataset layout expected by `train.py`

```
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ... (38 class folders)
└── valid/
    └── ... (38 class folders)
```

## ⚠️ Limitations

- One prediction per image – use a photo of **a single leaf**.
- Only the 38 classes above are known (there is no banana, for example).
- Trained on clear, close-up photos; photos with a busy background give lower accuracy.
- An educational tool – not a substitute for an agriculture expert.

## 🔮 Future scope

- Detect several leaves in one photo (object detection + classification)
- More crops and diseases, and real field images
- Transfer learning (MobileNet / EfficientNet) for higher accuracy
- Mobile app with TensorFlow Lite
- Treatment advice in Hindi and other languages
- Grad-CAM heat-maps to show which part of the leaf was used

## 🙏 Credits

- Dataset: PlantVillage, and the *New Plant Diseases Dataset* on Kaggle.
- The network design and notebooks are based on the public project
  [animesh1012/machineLearning](https://github.com/animesh1012/machineLearning),
  which was studied, reorganised and extended here (shared helper module, training and
  checking scripts, error handling).
- Libraries: TensorFlow / Keras, Streamlit, NumPy, Pillow, scikit-learn, Matplotlib.

## 👤 Author

**[Your Name]** – Department of Computer Applications, Graphic Era University, Dehradun
· [GitHub](https://github.com/<your-username>) · [LinkedIn](https://www.linkedin.com/in/<your-id>)

## 📄 License

This project is for educational purposes. Add a license of your choice (for example MIT) before reuse.# leaf_disease_pridection-using-machine-learning
