# Fine-tuning Faster R-CNN for Object Detection on Pascal VOC 2012

This project demonstrates how to fine-tune a Faster R-CNN model, pre-trained on the COCO dataset, for object detection on the Pascal VOC 2012 dataset. The model is adjusted to handle the 20 classes present in VOC, down from the 90 classes in COCO.

## Project Overview

This project aims to provide a practical example of fine-tuning a pre-trained Faster R-CNN model for a specific object detection task.  By leveraging a pre-trained model and adapting it to a new dataset, we can achieve effective object detection with limited computational resources.

**Key Features:**
### Key Components:
- **Faster R-CNN Backbone**: Initially uses MobileNet v3 and ResNet50 to experiment with lightweight and larger architectures.
- **Pascal VOC 2012 Dataset**: Custom loader for VOC dataset with 20 object categories.
- **Training & Evaluation**: Custom training script to evaluate the model and store the checkpoint with the highest mAP.

### Block Diagram:
```plaintext
   ┌──────────────────────┐
   │      Pascal VOC       │
   │    (20 Object Classes)│
   └─────────▲─────────────┘
             │
   ┌─────────▼─────────────┐
   │  Pre-trained Faster    │
   │      R-CNN Model       │
   │   (MobileNet v3/ResNet)│
   └─────────▲─────────────┘
             │
   ┌─────────▼─────────────┐
   │ Fine-tune Specific     │
   │ Layers & Output Heads  │
   └─────────▲─────────────┘
             │
   ┌─────────▼─────────────┐
   │ Training & Evaluation  │
   │ (Save Checkpoint with  │
   │  Best mAP)             │
   └───────────────────────┘
```
- **Fine-tuning:**  Adapts a pre-trained Faster R-CNN model for Pascal VOC 2012 dataset.
- **Multiple Backbones:** Explores performance with MobileNet v3 and ResNet50 backbones.
- **Resource-Aware Training:**  Demonstrates training on both Google Colab (limited resources) and Kaggle (enhanced resources) environments. 
- **Evaluation:** Provides scripts for testing the model on new images and reports mAP scores.

## Project Architecture

The project consists of two main implementations:

**1. Google Colab Implementation:**

- **`FRCNN_main.ipynb`:** Main Jupyter Notebook to execute the training and testing process.
- **`test_new_data.py`:** Script to test the trained model on new images.
- **`training.py`:** Contains the model definition, training loop, and saves the best model checkpoint based on mAP.
- **`voc_dataset_format.py`:**  Handles loading and preprocessing the Pascal VOC dataset. 

**2. Kaggle Implementation:**

- **`frcnn.ipynb`:**  Single Jupyter Notebook integrating training and evaluation using MobileNet v3 and ResNet50 backbones.

## Getting Started

### Prerequisites:

- Python 3.7+
- Required libraries: TensorFlow, Keras, OpenCV, etc. (refer to requirements.txt) 

### Running the Code:

**Google Colab:**

1.  Upload the project files to your Google Colab environment.
2.  Open `FRCNN_main.ipynb` and run all cells. 

**Kaggle:**

1.  Create a new Kaggle Notebook and upload the project files.
2.  Run the  `frcnn.ipynb.ipynb` notebook.

## Results

The achieved mAP (mean Average Precision) scores for different configurations are:

| Platform | Backbone          | Trainable Backbone Layers | Epochs | mAP     | Notes                                  |
| :-------- | :---------------- | :------------------------ | :----- | :------- | :------------------------------------ |
| Colab    | MobileNet v3      | 0                       | 2      | 0.0299  | Limited by resources                |
| Kaggle   | MobileNet v3      | 3                       | 10     | 0.1366  |                                       |
| Kaggle   | MobileNet v3      | 6                       | 10     | 0.0957  |                                       |
| Kaggle   | ResNet50          | 2                       | 5      | 0.0084  | Requires more epochs for better results |

## FAQ ❓
Q: Why is the mAP score so low?
A: Due to the limited computational resources on Colab, the model is trained for only a few epochs, resulting in lower mAP scores. On Kaggle, more epochs are used, and larger backbones are tested.

Q: How can I improve the performance?
A: You can try increasing the number of epochs, using a more powerful backbone, or fine-tuning more layers in the backbone.

Q: How do I load my custom dataset?**  
A: With **Colab**, push your image into your Google Drive, connect to the Drive, and insert the path into the test component in `FRCNN_main.ipynb`.  
For **Kaggle**, create your own dataset, upload your images, and paste the image path into the test component.

## Demo 🎬

See the demo video in the **Colab** folder and the prediction output of my model in the **Kaggle** folder.


## Future Work

- Experiment with different backbone architectures (e.g., EfficientNet, Inception).
- Optimize training parameters (learning rate, batch size) for improved performance.
- Implement data augmentation techniques to enhance model robustness. 
- Explore transfer learning from other object detection datasets.

## Contact

[![Facebook](https://img.shields.io/badge/Facebook-blue?style=for-the-badge&logo=Facebook&logoColor=white)](https://www.facebook.com/qq2719/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/quang-nv-ptit/)
