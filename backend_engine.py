# backend_engine.py
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class WaterPredictor:
    def __init__(self):
        # 1. 初始化时，自动加载大脑(.keras)和工具箱(.pkl)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = tf.keras.models.load_model(os.path.join(current_dir, 'water_model.keras'))
        self.artifacts = joblib.load(os.path.join(current_dir, 'model_artifacts.pkl'))
        
        self.scalers = self.artifacts['scalers']
        self.encoders = self.artifacts['encoders']
        self.wq_encoder = self.artifacts['water_quality_encoder']
        self.target_features = self.artifacts['target_features']
        self.site_data_count = self.artifacts.get('site_data_count', {})
        self.all_sites = self.artifacts.get('all_sites', [])
        self.LOOKBACK_DAYS = 3
        self.WARN_DATA_COUNT = 5

    def predict_any_site(self, site_history, target_point, target_year, target_quarter):
        """单站点单季度的核心预测逻辑"""
        site_count = self.site_data_count.get(target_point, 0)
        warning = f"警告：数据量过少（仅{site_count}条），结果为粗略外推" if site_count < self.WARN_DATA_COUNT else ""
        
        try:
            if site_count == 0 or len(site_history) == 0:
                raise ValueError("无历史数据")
                
            recent_data = site_history.tail(self.LOOKBACK_DAYS).copy()
            if len(recent_data) < self.LOOKBACK_DAYS:
                pad_df = pd.DataFrame([recent_data.iloc[0]] * (self.LOOKBACK_DAYS - len(recent_data)))
                recent_data = pd.concat([pad_df, recent_data], ignore_index=True)
                
            X_seq_input = recent_data[self.target_features].values.reshape(1, self.LOOKBACK_DAYS, len(self.target_features))
            
            spatial_features = ['sea_area_encoded', 'province_encoded', 'city_encoded', 'point_location_encoded', 'lon_norm', 'lat_norm']
            spatial_vals = recent_data.iloc[-1][spatial_features].values
            
            target_month = {1:4, 2:7, 3:10}[target_quarter]
            time_df = pd.DataFrame([[target_year, target_month, target_quarter]], columns=['year', 'month', 'quarter'])
            time_norm = self.scalers['time'].transform(time_df)[0]
            X_static_input = np.concatenate([spatial_vals, time_norm]).reshape(1, -1)
            
            pred_norm = self.model.predict([X_seq_input, X_static_input], verbose=0)[0]
            pred = self.scalers['target'].inverse_transform(pred_norm.reshape(1, -1))[0]
            
            quality_idx = max(0, min(int(round(pred[6])), len(self.wq_encoder.classes_) - 1))
            quality_label = self.wq_encoder.inverse_transform([quality_idx])[0]
            
            target_date = pd.to_datetime(f"{target_year}-{target_month}-01")
            
            return {
                '站点': target_point, '年份': target_year, '季度': target_quarter,
                '季度名称': {1:'spring', 2:'summer', 3:'autumn'}[target_quarter],
                '预测日期': target_date.strftime('%Y-%m-%d'),
                'pH': round(pred[0], 2), 'DO': round(pred[1], 2), 'COD': round(pred[2], 2),
                'IN': round(pred[3], 4), 'PO4': round(pred[4], 4), 'oil': round(pred[5], 3),
                'water_quality': quality_label, '数据量': site_count, '警告信息': warning,
                '_norm_preds': pred_norm 
            }
        except Exception as e:
            # 兜底逻辑
            fallback_norm = np.zeros(len(self.target_features))
            pred = self.scalers['target'].inverse_transform(fallback_norm.reshape(1, -1))[0]
            
            # 🛠️ 修正部分：先在外面把字典映射算好，再清清爽爽地塞进 f-string 里
            fallback_month = {1:4, 2:7, 3:10}[target_quarter]
            target_date = pd.to_datetime(f"{target_year}-{fallback_month}-01")
            
            return {
                '站点': target_point, '年份': target_year, '季度': target_quarter,
                '季度名称': {1:'spring', 2:'summer', 3:'autumn'}[target_quarter],
                '预测日期': target_date.strftime('%Y-%m-%d'),
                'pH': round(pred[0], 2), 'DO': round(pred[1], 2), 'COD': round(pred[2], 2),
                'IN': round(pred[3], 4), 'PO4': round(pred[4], 4), 'oil': round(pred[5], 3),
                'water_quality': "未知", '数据量': site_count,
                '警告信息': f"警告：预测异常，触发兜底", '_norm_preds': fallback_norm
            }
        
    def process_and_predict_all(self, df_raw):
        """
        供前端调用的主干接口：接收原始数据 -> 清洗 -> 批量预测 -> 返回 DataFrame
        """
        # 1. 基础数据清洗映射
        col_mapping = {'sea area':'sea_area', 'province':'province', 'city':'city', 'point location':'point_location', 'longitude':'longitude', 'latitude':'latitude', 'quarter':'quarter', 'year':'year', 'pH':'pH', 'Dissolved Oxygen (mg/L)': 'DO', 'chemical oxygen demand (mg/L)': 'COD', 'inorganic nitrogen (mg/L)': 'IN', 'active phosphate (mg/L)': 'PO4', 'oil type (mg/L)': 'oil', 'water quality':'water_quality'}
        df = df_raw.rename(columns=lambda x: col_mapping.get(x, x)).copy()
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2020)
        df['quarter'] = df['quarter'].map({'spring': 1, 'summer': 2, 'autumn': 3, '1':1, '2':2, '3':3, 1:1, 2:2, 3:3}).fillna(2)
        df['month'] = df['quarter'].map({1:4, 2:7, 3:10}).fillna(6)
        df['date'] = df.apply(lambda x: pd.to_datetime(f"{int(x['year'])}-{int(x['month'])}-01"), axis=1)
        df['time_idx'] = (df['date'] - df['date'].min()).dt.days.fillna(0)

        # 2. 特征编码与归一化
        for feat in ['sea_area', 'province', 'city', 'point_location']:
            if feat in df.columns:
                df[feat] = df[feat].fillna("未知").astype(str)
                known_classes = set(self.encoders[feat].classes_)
                df[feat] = df[feat].apply(lambda x: x if x in known_classes else self.encoders[feat].classes_[0])
                df[f'{feat}_encoded'] = self.encoders[feat].transform(df[feat])
            else:
                df[f'{feat}_encoded'] = 0

        df[['lon_norm', 'lat_norm']] = self.scalers['coord'].transform(df[['longitude', 'latitude']]) if 'longitude' in df.columns else self.scalers['coord'].transform([[110.0, 30.0]])
        
        for col in self.target_features[:-1]:
            df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
            
        df['water_quality'] = df.get('water_quality', "未知").fillna("未知").astype(str)
        known_wq = set(self.wq_encoder.classes_)
        df['water_quality'] = df['water_quality'].apply(lambda x: x if x in known_wq else self.wq_encoder.classes_[0])
        df['water_quality_encoded'] = self.wq_encoder.transform(df['water_quality']).astype(float)

        df[['year_norm', 'month_norm', 'quarter_norm']] = self.scalers['time'].transform(df[['year', 'month', 'quarter']])
        df[self.target_features] = self.scalers['target'].transform(df[self.target_features])

        # 3. 批量滚动预测
        years = range(2026, 2031)
        quarters = [1, 2, 3]
        all_predictions = []

        valid_sites = [s for s in self.all_sites if s in df['point_location'].unique()]
        
        for point in valid_sites:
            site_history = df[df['point_location_encoded'] == self.encoders['point_location'].transform([point])[0]].sort_values('time_idx').copy()
            
            for year in years:
                for q in quarters:
                    result = self.predict_any_site(site_history, point, year, q)
                    norm_preds = result.pop('_norm_preds')
                    all_predictions.append(result)
                    
                    if len(site_history) > 0:
                        new_row = site_history.iloc[-1].copy()
                        for i, feat in enumerate(self.target_features):
                            new_row[feat] = norm_preds[i]
                        time_norm = self.scalers['time'].transform([[year, {1:4, 2:7, 3:10}[q], q]])[0]
                        new_row['year_norm'], new_row['month_norm'], new_row['quarter_norm'] = time_norm
                        new_row['time_idx'] += 90 
                        site_history = pd.concat([site_history, pd.DataFrame([new_row])], ignore_index=True)

        return pd.DataFrame(all_predictions)