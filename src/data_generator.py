import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_users=2000, seed=42):
    """
    Генерирует реалистичные логи событий для SaaS-платформы аудио-плагинов.
    Заносит скрытые паттерны зависимости конверсии от онбординга и функций.
    """
    np.random.seed(seed)
    user_ids = [f"user_{i:04d}" for i in range(1, n_users + 1)]
    user_types = np.random.choice(['casual', 'professional'], size=n_users, p=[0.7, 0.3])
    
    events_data = []
    
    for uid, utype in zip(user_ids, user_types):
        # Базовая дата регистрации в январе 2026 года
        reg_date = datetime(2026, 1, 1) + timedelta(days=np.random.randint(0, 30), 
                                                   hours=np.random.randint(0, 24), 
                                                   minutes=np.random.randint(0, 60))
        
        # Шаг 1: Регистрация
        events_data.append({'user_id': uid, 'event': '1_registration', 'timestamp': reg_date})
        
        # Шаг 2: Скачивание инсталлятора (CR ~ 85%)
        if np.random.rand() < 0.85:
            t1 = reg_date + timedelta(minutes=int(np.random.exponential(15)))
            events_data.append({'user_id': uid, 'event': '2_download_installer', 'timestamp': t1})
            
            # Шаг 3: Первый запуск плагина в DAW (CR ~ 75% от скачавших)
            if np.random.rand() < 0.75:
                t2 = t1 + timedelta(hours=int(np.random.exponential(4)))
                events_data.append({'user_id': uid, 'event': '3_plugin_open', 'timestamp': t2})
                
                # Внутренняя активность в течение 14-дневного триала
                n_presets = np.random.randint(1, 15) if utype == 'casual' else np.random.randint(5, 40)
                n_exports = np.random.randint(0, 3) if utype == 'casual' else np.random.randint(1, 8)
                
                # Использование AI-мастеринг ассистента (Aha-момент)
                ai_clicked = np.random.choice([0, 1], p=[0.8, 0.2] if utype == 'casual' else [0.3, 0.7])
                
                current_time = t2
                
                # Переключение пресетов
                for _ in range(n_presets):
                    current_time += timedelta(minutes=np.random.randint(2, 45))
                    if current_time < reg_date + timedelta(days=14):
                        events_data.append({'user_id': uid, 'event': '4_preset_change', 'timestamp': current_time})
                
                # Использование AI
                if ai_clicked:
                    current_time += timedelta(minutes=np.random.randint(5, 20))
                    if current_time < reg_date + timedelta(days=14):
                        events_data.append({'user_id': uid, 'event': '5_ai_assistant_click', 'timestamp': current_time})
                
                # Экспорт финального трека
                for _ in range(n_exports):
                    current_time += timedelta(days=np.random.randint(1, 4), hours=np.random.randint(-6, 6))
                    if current_time < reg_date + timedelta(days=14):
                        events_data.append({'user_id': uid, 'event': '6_audio_export', 'timestamp': current_time})
                
                # Логика конверсии в платную подписку
                base_prob = 0.02 if utype == 'casual' else 0.08
                if ai_clicked: 
                    base_prob += 0.18  # Весомый вклад Aha-момента
                if n_exports >= 3: 
                    base_prob += 0.22  # Вклад регулярного использования
                    
                # Ограничиваем максимальную вероятность
                final_prob = min(base_prob, 0.85)
                
                if np.random.rand() < final_prob and current_time < reg_date + timedelta(days=14):
                    t_sub = current_time + timedelta(hours=np.random.randint(1, 48))
                    events_data.append({'user_id': uid, 'event': '7_subscription_purchase', 'timestamp': t_sub})
                    
    df_events = pd.DataFrame(events_data)
    df_events = df_events.sort_values('timestamp').reset_index(drop=True)
    return df_events

if __name__ == "__main__":
    print("Генерация данных...")
    df = generate_synthetic_data(n_users=2000)
    
    # Поднимаемся на уровень выше, так как скрипт лежит в src/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'events_log.csv')
    df.to_csv(file_path, index=False)
    print(f"Успешно сгенерировано {len(df)} событий. Файл сохранен в 'data/events_log.csv'")