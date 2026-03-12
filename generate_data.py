import pandas as pd
import numpy as np
from pathlib import Path

# Створюємо директорію, якщо її немає
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Налаштування
np.random.seed(42)
n_samples = 500

# Генерація ознак (features)
data = {
    'feature1': np.random.randn(n_samples),      # Числова ознака
    'feature2': np.random.rand(n_samples) * 100, # Числова ознака
    'feature3': np.random.normal(10, 2, n_samples), # Числова ознака
    'category_col': np.random.choice(['A', 'B', 'C'], n_samples) # Категорійна ознака
}

df = pd.DataFrame(data)

# Створення цільової змінної (target) для регресії 
# target = 2*f1 + 0.5*f2 - 1.2*f3 + шум
df['target'] = (2 * df['feature1'] + 
                0.5 * df['feature2'] - 
                1.2 * df['feature3'] + 
                np.random.normal(0, 1, n_samples))

# Збереження
df.to_csv("data/raw/variant08.csv", index=False)
print("Файл data/raw/variant08.csv успішно створено!")