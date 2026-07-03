import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

def calculate_funnel_metrics(df_events, total_users):
    """
    Вычисляет классические метрики продуктовой воронки.
    """
    funnel = df_events.groupby('event')['user_id'].nunique().reset_index()
    funnel.columns = ['Step', 'Unique_Users']
    funnel = funnel.sort_values('Step').reset_index(drop=True)
    
    funnel['CR_to_Base_%'] = round((funnel['Unique_Users'] / total_users) * 100, 2)
    funnel['Conversion_from_Previous_%'] = round(funnel['Unique_Users'].pct_change().add(1).fillna(1) * 100, 2)
    funnel['Drop_off_Rate_%'] = 100 - funnel['Conversion_from_Previous_%']
    funnel.loc[0, 'Drop_off_Rate_%'] = 0.0
    
    return funnel

def prepare_user_features(df_events):
    """
    Агрегирует логи до уровня профилей пользователей для регрессионного анализа.
    """
    active_users = df_events[df_events['event'] == '3_plugin_open']['user_id'].unique()
    user_features = []
    
    for uid in active_users:
        user_logs = df_events[df_events['user_id'] == uid]
        
        presets_count = len(user_logs[user_logs['event'] == '4_preset_change'])
        ai_clicked = int(any(user_logs['event'] == '5_ai_assistant_click'))
        exports_count = len(user_logs[user_logs['event'] == '6_audio_export'])
        converted = int(any(user_logs['event'] == '7_subscription_purchase'))
        
        user_features.append({
            'user_id': uid,
            'presets_changed': presets_count,
            'ai_assistant_used': ai_clicked,
            'audio_exports': exports_count,
            'converted': converted
        })
        
    return pd.DataFrame(user_features)

def estimate_logistic_regression(df_features):
    """
    Строит модель логистической регрессии для поиска статистически значимых факторов.
    """
    X = df_features[['presets_changed', 'ai_assistant_used', 'audio_exports']]
    X = sm.add_constant(X)
    y = df_features['converted']
    
    model = sm.Logit(y, X).fit(disp=0)
    return model

def run_z_test(success_A, nobs_A, success_B, nobs_B):
    """
    Выполняет Z-тест равенства долей для двух независимых выборок (A/B тест).
    """
    count = np.array([success_B, success_A])
    nobs = np.array([nobs_B, nobs_A])
    
    stat, p_value = proportions_ztest(count, nobs, alternative='larger')
    return stat, p_value