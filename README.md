# 🚀 CAREER RECOMMENDATION SYSTEM

The Career Recommendation System is designed to help students make informed career choices by analyzing their personal attributes, interests, and other key factors. The system employs machine learning algorithms to generate personalized career recommendations based on user data collected from the questionnaire.

---

## SYSTEM WORKFLOW

The system follows a structured workflow consisting of the following key processes:

### Data Collection
- The system gathers user data through a series of questions that focus on personality type, skills, interests, and other relevant factors. Once the data is collected, it is sorted and processed using our recommendation algorithm to ensure accurate and meaningful career suggestions.

### Recommendation Model
- Our model uses the RandomForestClassifier, as it achieved the highest accuracy among our evaluated algorithms. By applying RandomForestClassifier, the system effectively sorts and classifies user data, allowing it to generate career recommendations tailored to the user’s responses.

### Expected Output
- Based on the collected information, the system predicts a suitable career path for the user, providing a recommendation that aligns with their profile and preferences.

---

## 🌟 FEATURES
✔️ **User-Friendly GUI** built with Swing / JavaFX  
✔️ **Machine Learning Model** using [Scikit-Learn, Pandas, NumPy]  
✔️ **Abstracted Implementation** for developers to modify and extend  
✔️ **Real-time Visualization** of predictions

---

## 🛠️ INSTALLATION
1. Clone the repository:  
   ```bash
   git clone https://github.com/shwarmanism/CareerRecommendation.git
   cd CareerRecommendation
2. Install required dependencies
   ```bash
   pip install -r requirements.txt
3. Run the application
   ```bash
   java app.java
---

## 🎨 GUI USAGE
   IKAW NA DITO RALPH
  
---

## ⚙️ Usage (Abstract Implementation)
   In the implementation of abstraction, we utilized the abstraction by using it to categorize the different fields. starting with the abstract class CareerField, this sets up
   the format of the rest of the subclasses with field(), recommendationCourse(), jobOpportunities(), and otherInfo() as its abstract methods. each field subclass then has their 
   information placed into these methods, by having them inherit CareerField. a career field is then selected using the switch statement to display its information for the user
   when it makes its recommendation.

---

## 📊 DATASET
- Name: train_data, test_data(Sum of user prompts)
- Source: https://docs.google.com/spreadsheets/d/1OlIQXwl1kYaHx4sPqv8HoqTcQg5o-mhQKOK3A95mjKs/edit?usp=sharing
- Format: .CSV
- Target Variable: Target_Field
  
## 🗝️ DATASET FEATURES DESCRIPTION
- Age (int) – The age of the individual.
- Strand (object) – The academic track or strand the student is enrolled in (e.g., STEM, HUMSS, TVL, ABM, etc.).
- MBTI (object) – The Myers-Briggs Type Indicator (e.g., INTP, ESFP), representing the individual's personality type.
- Extracurricular (object) – The extracurricular activities the individual participates in (e.g., Debate, Art Club, Science Club).
- Personality_Type (object) – A general classification of personality (e.g., Introvert, Extrovert).
- Study_Hours (float) – The average number of hours the individual spends studying.
- Tech_Savvy_Score (int) – A score representing the individual's technological proficiency.
- Creativity_Score (int) – A score indicating the individual's creativity level.
- Family_Income_Bracket (object) – The socioeconomic status of the individual’s family (e.g., Low, Middle, High).
- Cultural_Influence (object) – The primary source of cultural influence on the individual (e.g., Family, Society, Personal Interest).
- Scholarship_Status (object) – Whether the individual has a scholarship (Yes/No).
- Primary_Skill (object) – The individual's main skill (e.g., Critical Thinking, Communication, Leadership).
- Hobby (object) – The individual's primary hobby (e.g., Cooking, Sports, Photography, Traveling).
- Future_Field_Security (object) – The perceived job security of the individual's target field (Low, Moderate, High).
- Work_Flexibility (object) – The level of work flexibility preferred by the individual (e.g., Rigid, Flexible).
- Target_Field (object) – The career field the individual aims to enter (e.g., Law and Legal Services, Public Service, Business/Commerce). <- Target Variable

## 📈 Accuracy and Metrics Performance
- Best Classification Algorithm: RandomForestClassifier
  
| Metric         |     Value    |
|----------------|--------------|
| Accuracy       |est. 69% - 63%|
| Macro avg      |est. 66% - 60%|
| Weighted avg   |est. 68% - 61%|

---

## CONTACTS
| **Component**           | **Author**          | **Email**                |
|--------------------------|---------------------|--------------------------|
| Abstraction Implementation | Lucius Gamboa       | luciusgamboa03@gmail.com       |
| GUI Implementation      | Ralph Wendel Fortus       | ralphwendelf@gmail.com       |
| ML Model                | Nhico Paragas       | nj.paragas8@gmail.com      |




