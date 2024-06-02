
# PetVision Detector

PetVision is a deep learning project that utilizes a Convolutional Neural Network (CNN) to distinguish between images of cats and dogs. Built with TensorFlow, this project demonstrates the application of image preprocessing, model training, and model evaluation with additional insights into the model's decision-making process through feature map visualization.

## Project Setup

### Main Dependencies
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Matplotlib
- NumPy

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/muneebalichishti01/PetVision-Detector
cd PetVision-Detector
```

Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage
To train the model from scratch, run:
```bash
python convolutional_neural_network.py
```

## Using the already trained model
The trained model is saved in the `model` directory. To use the trained model for predictions, run the following command:
```bash
python "Predict using trained model.py"
```

### Model Architecture
The CNN model used in this project includes the following layers:
- Conv2D with 32 filters
- MaxPooling2D
- Conv2D with 32 filters
- MaxPooling2D
- Flatten
- Dense with 128 units
- Output Dense with sigmoid activation

### Visualizations
This project includes visualizations for:
- Training and validation loss and accuracy
- Feature maps from the convolutional layers
- Predictions with bounding box annotations on images

## Results
Discuss the accuracy, precision, and recall obtained from the model, and include some example predictions with images displayed in the README.

## Future Work
Possible extensions or improvements to the project, such as implementing additional layers, experimenting with different architectures, or tackling more complex image classification tasks.

## License
This project is open source and available under the [MIT License](LICENSE.md).

## Acknowledgments
Dataset used in this project is from the Deep Learning A-Z 2024, Udemy Course and can be [downloaded here](https://drive.google.com/file/d/1M3PDpOpVz6OQBZWuBMJfzpAm3l-9OrVv/view?usp=sharing).
