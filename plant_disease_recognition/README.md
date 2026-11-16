# Plant Disease Recognition System

Upload a photo of a plant leaf; a CNN trained on the *New Plant Diseases Dataset*
(PlantVillage, 87K images, 38 classes) tells you the plant, the disease (or "healthy"),
how sure it is, and the top 3 possibilities. Built with TensorFlow + Streamlit.

Based on the "Plant Disease Prediction" project by animesh1012
(github.com/animesh1012/machineLearning). Credit them if you present this.

## What is in this folder
| File | What it does |
|---|---|
| `main.py` | The web app (Home / About / Disease Recognition pages) |
| `plant_labels.py` | The 38 class names, loads the model, predicts one image |
| `train.py` | Trains the model from the command line |
| `Train_plant_disease.ipynb` | Same training, as a notebook (with accuracy graph + confusion matrix) |
| `Test_plant_disease.ipynb` | Tests the model on one image, step by step |
| `sample_images/rust_leaf.jpg` | A leaf photo to try |
| `requirements.txt` | The packages to install |

## Step 1. One-time setup (Windows)
1. Install **Python 3.11** (python.org). Tick **"Add python.exe to PATH"**. Newer versions
   (3.12, 3.13) do NOT work with TensorFlow 2.15.
2. Open this folder in VS Code (File, Open Folder) and open the terminal (Ctrl + `).
3. Create and activate the virtual environment:
   ```powershell
   py -3.11 -m venv venv
   venv\Scripts\activate
   python --version
   ```
   The last line must say `Python 3.11.x`. If PowerShell says scripts are disabled, run
   `Set-ExecutionPolicy -Scope Process Bypass` and activate again.
4. Install the packages (always `python -m pip`, never plain `pip`):
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   This takes several minutes (TensorFlow is large).

## Step 2. Get the trained model (pick ONE)
The app needs a file called **`trained_plant_disease_model.keras`** next to `main.py`
(`trained_model.keras` also works).

**A. Download it (fastest).** If someone shared a trained model with you, copy the file
into this folder.

**B. Train it yourself.**
1. Download the dataset from Kaggle: search **"New Plant Diseases Dataset"** by *vipoooool*
   (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and extract it.
2. Find the folder that directly contains `train` and `valid`
   (usually `New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)`).
3. Run (the `1` is the number of epochs; start small):
   ```powershell
   python train.py "C:\datasets\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)" 1
   ```
   This model is large: on a CPU even 1 epoch can take a long time, and 10 epochs
   (about 94% accuracy) can take many hours. A GPU or Google Colab is much faster.
   Or use `Train_plant_disease.ipynb`, which does the same and draws the graphs.

## Step 3. Run the app
```powershell
python -m streamlit run main.py
```
A browser tab opens. Choose **Disease Recognition**, upload a leaf photo (try
`sample_images/rust_leaf.jpg`), and press **Predict**. Stop the app with Ctrl+C.

Next time you only need: open the folder, `venv\Scripts\activate`, then the run command.

## Troubleshooting
- **"blocked by an Application Control policy" when running `pip`**: use
  `python -m pip ...` instead of `pip ...`.
- **`No trained model found`** shown in the app: the model file is missing or misnamed (Step 2).
- **`could not load it`**: the model was saved with a newer TensorFlow than 2.15. Train it
  yourself (Step 2B) or install the same TensorFlow version that saved it.
- **`ModuleNotFoundError`**: the venv is not active (you do not see `(venv)`), or Step 1.4 did not finish.
- **Wrong answers on your own photos**: use one leaf, close up, plain background.
  The model only knows the 14 plants of the dataset (no banana).
- To run the tests of the notebooks: open them in VS Code and pick the `venv` Python as kernel.
