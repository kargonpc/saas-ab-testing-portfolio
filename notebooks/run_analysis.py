# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем корень проекта в пути, чтобы импорты из src работали корректно
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.data_generator import generate_synthetic_data
from src.stats_utils import calculate_funnel_metrics, prepare_user_features, estimate_logistic_regression, run_z_test

# Установка стилей графиков
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

print("=== ШАГ 1: Загрузка или генерация данных ===")
data_file = os.path.join(base_dir, 'data', 'events_log.csv')
if not os.path.exists(data_file):
    df_events = generate_synthetic_data(n_users=2000)
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    df_events.to_csv(data_file, index=False)
else:
    df_events = pd.read_csv(data_file)
print(f"Загружено событий: {len(df_events)}")

print("\n=== ШАГ 2: Анализ продуктовой воронки ===")
funnel = calculate_funnel_metrics(df_events, total_users=2000)
print(funnel.to_string(index=False))

# Визуализация воронки
plt.figure(figsize=(12, 6))
sns.barplot(x='Unique_Users', y='Step', data=funnel, palette='Blues_r')
for index, value in enumerate(funnel['Unique_Users']):
    plt.text(value, index, f" {value} (CR: {funnel.loc[index, 'CR_to_Base_%']}% )", va='center', fontweight='bold')
plt.title('Продуктовая воронка активации', fontsize=14)
plt.xlabel('Количество уникальных пользователей')
plt.ylabel('Этап воронки')
plt.tight_layout()

chart_path = os.path.join(base_dir, 'notebooks', 'funnel_chart.png')
plt.savefig(chart_path)
print(f"График воронки сохранен в {chart_path}")

print("\n=== ШАГ 3: Поиск Aha-момента (Логистическая регрессия) ===")
df_features = prepare_user_features(df_events)
model = estimate_logistic_regression(df_features)
print(model.summary2().tables[1])

print("\n=== ШАГ 4: Расчет и анализ результатов A/B-теста ===")
# Моделируем эксперимент изменения онбординга
np.random.seed(42)
n_ab = 1500
conversion_A = 0.075
conversion_B = 0.110

success_A = np.random.binomial(n_ab, conversion_A)
success_B = np.random.binomial(n_ab, conversion_B)

stat, p_val = run_z_test(success_A, n_ab, success_B, n_ab)

print(f"Группа A (Контроль): Выборка={n_ab}, Конверсии={success_A}, CR={success_A/n_ab:.4%}")
print(f"Группа B (Тест):    Выборка={n_ab}, Конверсии={success_B}, CR={success_B/n_ab:.4%}")
print(f"Z-статистика: {stat:.4f}")
print(f"p-value: {p_val:.5f}")

if p_val < 0.05:
    print("Результат: Отклоняем нулевую гипотезу! Эффект статистически значим. Внедряем новый онбординг.")
else:
    print("Результат: Не удалось отклонить нулевую гипотезу. Эффект отсутствует.")