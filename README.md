
# 🌸 Flower Classification Web App 🌼

Welcome to the **Flower Classification Web App**! This project demonstrates a simple yet effective image classification system using **Streamlit** and **TensorFlow**. 🌟

## 🌟 Features
- Classifies images of flowers into **five categories**:
    1. 🌼 Daisy
    2. 🌻 Dandelion
    3. 🌹 Rose
    4. 🌞 Sunflower
    5. 🌷 Tulip
- User-friendly interface powered by **Streamlit**.
- Pre-trained TensorFlow model for accurate predictions.

## 🚀 Quick Start Guide
1. Clone this repository:
     ```bash
     git clone https://github.com/your-repo/flower-classification.git
     ```
2. Navigate to the project directory:
     ```bash
     cd flower-classification
     ```
3. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
4. Run the app locally:
     ```bash
     streamlit run app.py
     ```
5. Open the app in your browser (default URL: `http://localhost:8501/`).

## 🖼️ How to Use
1. Click on the **Browse files** button to upload an image of a flower. 🌸
2. The app will process the image and classify it into one of the five categories.
3. View the results along with the confidence score. 🎉

## 🛠️ Technical Details
- **Model**: A TensorFlow-based flower classification model trained on a subset of the [Flowers Dataset](https://www.tensorflow.org/datasets/catalog/flowers).
- **Weights**: The trained model weights are stored in `flower_model_trained.hdf5`.
- **Code**:
    - Model training and modification: `model.py`
    - Web app implementation: `app.py`

## 📊 Sample Outputs
### Home Page
<img src='misc/sample_home_page.png' width=700 alt="Sample Home Page">

### Classification Result
<img src='misc/sample_output.png' width=700 alt="Sample Output">

## 📚 References
- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Streamlit Documentation](https://docs.streamlit.io/en/stable/)

## 💡 Future Enhancements
- Add support for more flower categories. 🌺
- Improve the model's accuracy with additional training data. 📈
- Deploy the app on a cloud platform for public access. ☁️

Enjoy exploring the beauty of flowers with AI! 🌸✨
# Image-Classification-Streamlit-TensorFlow
A basic web-app for image classification using Streamlit and TensorFlow.

It classifies the given image of a flower into one of the following five categories :-  
1. Daisy
2. Dandelion
3. Rose
4. Sunflower
5. Tulip

## Commands

To run the app locally, use the following command :-  
`streamlit run app.py`  

The webpage should open in the browser automatically.  
If it doesn't, the local URL would be output in the terminal, just copy it and open it in the browser manually.  
By default, it would be `http://localhost:8501/`  

Click on `Browse files` and choose an image from your computer to upload.  
Once uploaded, the model will perform inference and the output will be displayed.  

## Output

<img src ='misc/sample_home_page.png' width = 700>  

<img src ='misc/sample_output.png' width = 700>


## Notes
* A simple flower classification model was trained using TensorFlow.  
* The weights are stored as `flower_model_trained.hdf5`.  
* The code to train the modify and train the model can be found in `model.py`.  
* The web-app created using Streamlit can be found in `app.py`


## References

* https://www.tensorflow.org/tutorials/images/classification
* https://docs.streamlit.io/en/stable/
