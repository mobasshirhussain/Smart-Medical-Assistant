## Smart-Medical-Assistant
## INTRODUCTION
 This project aims to develop a Smart Health Assistant capable of predicting 
diseases based on user-provided symptoms using machine learning techniques. In 
real-life situations, individuals often face difficulty in identifying their illness when 
experiencing multiple symptoms simultaneously. To address this challenge, the 
system analyzes the relationship between 132 symptoms and 42 different diseases 
to generate accurate predictions. Each symptom is represented in a binary format 
(1 = present, 0 = absent), allowing the model to effectively map symptoms to their 
respective diseases.
 The primary objective of the project is to deliver quick, reliable, and accessible 
disease predictions, while also suggesting the appropriate doctor specialization for 
consultation. A user-friendly interface will be developed where users can simply 
select their symptoms and instantly receive results.
 By integrating symptom analysis, machine learning-based prediction models, and a 
doctor recommendation system, this Smart Health Assistant will serve as a 
decision-support tool. It will not only help individuals gain an early understanding 
of their health condition but also encourage timely medical consultation, thereby 
improving healthcare accessibility and awareness. 

## SPECIFIC OBJECTIVES
 ➢Develop a Predictive Model: To train a RandomForestClassifier model on a 
symptom dataset for accurate disease classification.
 ➢Create a User-Friendly Interface: To build a simple and interactive web 
application using Streamlit, allowing users to easily select their symptoms.
 ➢Provide Doctor Recommendations: To integrate a pre-defined mapping of 41 
different diseases to medical specialists and suggest the correct one based on 
the prediction.
 ➢Evaluate and Visualize: To assess the model's performance using metrics like 
accuracy and precision, and to identify and visualize the most important features 
(symptoms).

## PROJECT USECASES:
#### Use Case 1: 
Decision Support for Doctors - Saves time in understanding 
patient’s condition and speeds up treatment.
#### Use Case 2: 
Educational Use Case - Useful for medical students and 
researchers to understand symptom-disease relationships.

## SCOPE OF PROJECT:
The system predicts diseases based on user symptoms using machine learning 
models.
It provides possible disease outcomes and recommends suitable specialist 
doctors.
Designed for general users (self-assessment) and health professionals (decision 
support).
Limited to diseases available in the dataset; not a replacement for medical 
diagnosis.
Future scope: integrate with real patient data, IoT health devices, multi
language support, and telemedicine platforms.

## METHODOLOGY
 1. Data Preparation: Loaded the Training.csv data and cleaned it using Pandas.
 2. Encoding and Scaling:
 • The target variable (prognosis) was converted into numerical labels with 
LabelEncoder.
 • The feature data (symptoms) was normalized using StandardScaler.
 3. Train-Test Split: The dataset was split into 80% training and 20% testing sets, 
using stratify to ensure proportional representation of all diseases.
 4. Model Training: A RandomForestClassifier with 300 estimators was trained on 
the training data.
 5. Model Serialization: The trained model, scaler, and encoder were saved into a 
single 
6. medical_model.pkl file using pickle for use in the web app.

 ## Dataset
 The dataset used is “Trainning.csv”, which contains:
 -- 132 input features (symptoms): Each symptom is represented in 
binary format (0 = Absent, 1 = Present).
 Example: itching, skin_rash, cough, fever, headache, fatigue, 
nausea, chest_pain, abdominal_pain, weight_loss etc.
-- 1 target feature (Disease Prediction): Indicates the predicted 
disease (e.g., Fungal infection, Allergy, Diabetes, Dengue, GERD, 
Chronic cholestasis).
-- Dataset Shape: 42 rows × 133 columns.
-- Data Type: curated datasets for research based, Categorical 
(binary values for symptoms, categorical string for disease).

## 100% accuracy is expected because:
 • Each disease has a fixed set of symptom mappings.
 • The ML model efficiently learns these mappings.
### Research-based project with a curated dataset (knowledge base, not real patient data).


 # IMPACT:
 Health Awareness: Helps in making users more aware of their health symptoms.
 Guidance for Care: Guides people toward seeking the right type of specialist.
 Educational Value: Demonstrates how AI can be effectively used to build assistive 
tools in healthcare.


