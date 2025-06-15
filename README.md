# Прогнозирование диабета с помощью ML

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-green)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85.0-blue)](https://fastapi.tiangolo.com)

Проект по предсказанию диабета у пациентов с использованием логистической регрессии и развертыванием модели как REST API на AWS EC2.


## 🔍 О проекте
Полный ML pipeline для бинарной классификации:
1. EDA и предобработка данных
2. Обучение модели логистической регрессии
3. Создание REST API на FastAPI
4. Деплой на облачном сервере AWS EC2

**Цель**: Спрогнозировать вероятность диабета у пациенток на основе медицинских показателей.

## 📊 Набор данных
Используется классический [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database):
- 768 наблюдений
- 8 медицинских признаков
- Бинарный таргет (1 - диабет, 0 - здоров)

**Особенности данных**:
```python
['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
