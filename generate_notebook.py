import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# YOLO/CNN Practical Assignment\n",
                "## Tasks 1-7, 10 & 11: Training a custom Nano model using API datasets\n",
                "---\n",
                "Run this notebook cell-by-cell. It will download the COCO128 dataset implicitly and train a `yolov8n.pt` model."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Install Required Packages"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "!{sys.executable} -m pip install ultralytics Flask pillow requests matplotlib opencv-python"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Task 1 & 2: Load Dataset API & Verify Images/Labels"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ultralytics import YOLO\n",
                "import os\n",
                "from pathlib import Path\n",
                "import matplotlib.pyplot as plt\n",
                "import matplotlib.image as mpimg\n",
                "\n",
                "# Initialize the nano model (Task 1)\n",
                "model = YOLO('yolov8n.pt')\n",
                "print('✅ YOLOv8 Nano model initialized!')\n",
                "\n",
                "dataset_dir = Path(os.path.expanduser('~')) / 'datasets' / 'coco128'\n",
                "if dataset_dir.exists():\n",
                "    print('Dataset already found locally. If not found, it downloads automatically at Task 3.')\n",
                "    img_path = list((dataset_dir / 'images' / 'train2017').glob('*.jpg'))[:2]\n",
                "    if img_path:\n",
                "        fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
                "        for i, img in enumerate(img_path):\n",
                "            axes[i].imshow(mpimg.imread(img))\n",
                "            axes[i].axis('off')\n",
                "            axes[i].set_title(f'Sample {i+1}')\n",
                "        plt.show()\n",
                "else:\n",
                "    print('Dataset will be auto-downloaded on first train trigger.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Task 3 & 4: Train YOLO (Custom Epochs & Batch Size)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Training with epoch=5 and batch=16\n",
                "results = model.train(\n",
                "    data='coco128.yaml',\n",
                "    epochs=5,\n",
                "    imgsz=640,\n",
                "    batch=16,\n",
                "    project='Student_YOLO',\n",
                "    name='run1',\n",
                "    plots=True\n",
                ")\n",
                "print('✅ Training Finished!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Task 5: Validate Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "val_metrics = model.val()\n",
                "print(f'mAP50: {val_metrics.box.map50:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Task 6: Test Model with model.predict()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ultralytics import YOLO\n",
                "import cv2\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "import os\n",
                "\n",
                "# Reload model explicitly to avoid NameError if Jupyter restarted\n",
                "model_path = 'best.pt'\n",
                "if not os.path.exists(model_path):\n",
                "    run_dirs = [d for d in Path('runs/detect/Student_YOLO').iterdir() if d.is_dir() and d.name.startswith('run')]\n",
                "    if run_dirs:\n",
                "        latest_run = max(run_dirs, key=os.path.getmtime)\n",
                "        model_path = str(latest_run / 'weights' / 'best.pt')\n",
                "try:\n",
                "    model = YOLO(model_path)\n",
                "    # Predict directly from online test image\n",
                "    results = model('https://ultralytics.com/images/bus.jpg')\n",
                "    # YOLO automatically renders the box mapping for us inside its array\n",
                "    annotated_frame = results[0].plot()\n",
                "    # Display the rendering directly without touching the file system\n",
                "    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))\n",
                "    plt.axis('off')\n",
                "    plt.show()\n",
                "except Exception as e:\n",
                "    print(f'Error: {e} - You need to train the model first!')"
            ]
        },
        {
             "cell_type": "markdown",
             "metadata": {},
             "source": [
                 "---\n",
                 "### Task 7: Save best.pt"
             ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "import os\n",
                "\n",
                "# Dynamically find the latest run folder\n",
                "base_dir = Path('runs/detect/Student_YOLO')\n",
                "run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('run')]\n",
                "if run_dirs:\n",
                "    latest_run = max(run_dirs, key=os.path.getmtime)\n",
                "    source = latest_run / 'weights' / 'best.pt'\n",
                "    dest = Path('best.pt')\n",
                "    if source.exists():\n",
                "        shutil.copy(source, dest)\n",
                "        print(f'✅ successfully saved best model to {dest.absolute()}')\n",
                "    else:\n",
                "        print(f'best.pt not found inside {source}')\n",
                "else:\n",
                "    print('No training run folders found.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("CNN_YOLO_Assignment_Fresh.ipynb", "w") as f:
    json.dump(notebook, f, indent=4)
