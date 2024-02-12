import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

# Загрузка данных
data = pd.read_csv("realest.csv")

# Замена пропущенных значений символом None
data.replace("NA", np.nan, inplace=True)

# Перевод всех значений в числовой формат
data = data.apply(pd.to_numeric, errors='ignore')

# Функция для восстановления пропущенных значений и анализа закономерностей с помощью нейронной сети
def fill_missing_values_and_analyze(df):
    # Копируем датафрейм, чтобы не изменять оригинальные данные
    filled_df = df.copy()
    
    # Проходим по каждому столбцу
    for column in filled_df.columns:
        # Проверяем, есть ли пропущенные значения в столбце
        if filled_df[column].isnull().any():
            # Разделяем данные на признаки и целевую переменную
            train_data = filled_df.dropna(subset=[column])
            X_train = train_data.drop(column, axis=1)
            y_train = train_data[column]
            
            # Создаем модель нейронной сети
            model = Sequential()
            model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            
            # Обучаем модель на данных с пропущенными значениями в качестве целевой переменной
            model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
            
            # Предсказываем пропущенные значения
            for index, row in filled_df[filled_df[column].isnull()].iterrows():
                X_pred = row.drop(column).values.reshape(1, -1)
                filled_df.at[index, column] = model.predict(X_pred)
    
    # Анализируем данные и даем советы
    # Здесь вы можете добавить любую логику анализа данных и формулирования советов
    
    return filled_df

# Визуализация данных до восстановления
def visualize_data_before(data):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(data.columns):
        ax = axes[i]
        ax.hist(data[col].dropna(), bins=20, color='skyblue', edgecolor='black')
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        
        # Добавляем информацию о пропущенных значениях
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            ax.annotate(f'Missing: {missing_count}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10, color='red')
    
    # Убираем пустые subplot'ы
    for i in range(len(data.columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# Визуализация данных до восстановления
visualize_data_before(data)

# Восстановление пропущенных значений и анализ данных
filled_data = fill_missing_values_and_analyze(data)

# Визуализация данных после восстановления
visualize_data_before(filled_data)
