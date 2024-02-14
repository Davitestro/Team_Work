import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv("realest.csv")

data.replace("NA", np.nan, inplace=True)

data = data.apply(pd.to_numeric, errors='ignore')

def fill_missing_values_and_analyze(df):
    filled_df = df.copy()
    
    for column in filled_df.columns:
        if filled_df[column].isnull().any():
            train_data = filled_df.dropna(subset=[column])
            X_train = train_data.drop(column, axis=1)
            y_train = train_data[column]
            
            model = Sequential()
            model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            
            model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
            
            for index, row in filled_df[filled_df[column].isnull()].iterrows():
                X_pred = row.drop(column).values.reshape(1, -1)
                filled_df.at[index, column] = model.predict(X_pred)
    

    
    return filled_df

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
        
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            ax.annotate(f'Missing: {missing_count}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10, color='red')
    
    for i in range(len(data.columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

visualize_data_before(data)

filled_data = fill_missing_values_and_analyze(data)

print(f"\t\t\tHead\n\n\n{filled_data.head()}\n\n")
print(f"\t\t\tDescription\n\n\n{filled_data.describe()}\n\n")
print(f"\t\t\tinfo\n\n\n{filled_data.info()}\n\n")


visualize_data_before(filled_data)


corr_matrix = data.corr()

plt.figure(figsize=(12, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.title('Correlation Matrix')
plt.colorbar()

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)

plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(data['Space'], data['Price'], alpha=0.7, color='blue')
plt.title('Scatter Plot of Price vs Space')
plt.xlabel('Space')
plt.ylabel('Price')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
data.boxplot(column=['Price', 'Space', 'Lot'])
plt.title('Box Plot of Price, Space, and Lot')
plt.ylabel('Value')
plt.show()