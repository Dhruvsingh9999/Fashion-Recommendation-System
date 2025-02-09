
# 🛍 Fashion Recommendation System

## 🚀 Overview
This project is an AI-powered **Fashion Recommendation System** that suggests similar fashion items based on image inputs. It leverages **Deep Learning (ResNet50)** for feature extraction and **Nearest Neighbors Algorithm** for recommendations.

## 🏠 Features
- Upload an image of clothing or accessories.
- Extracts features using a pre-trained **ResNet50 model**.
- Recommends **top 5 most similar fashion items**.
- Interactive **Streamlit web app**.
  
## ⚙️ Tech Stack
- **Python**
- **TensorFlow & Keras (ResNet50)**
- **Scikit-learn (Nearest Neighbors)**
- **NumPy, PIL**
- **Streamlit (UI)**

## 👤 File Structure
- `main.py` - Streamlit web app for image upload and recommendation.
- `features_extractor.py` - Extracts image features using ResNet50.
- `embeddings.pkl` - Precomputed feature embeddings.
- `filenames.pkl` - List of image filenames.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-recommendation.git
   cd fashion-recommendation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run main.py
   ```

## 🎯 Usage
- Upload an image of clothing.
- Get **AI-generated recommendations** for similar items.
- Explore the **top 5 recommended fashion products**.

## 📌 Future Enhancements
- Improve recommendation accuracy with a fine-tuned model.
- Support for multiple fashion categories.
- Integrate price and brand filters.

🚀 **Try it out and let us know your feedback!**

