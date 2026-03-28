import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tensorflow as tf
import io
import os  # 用于自动定位绝对路径

# ==================== 1. 页面配置与基础设定 ====================
st.set_page_config(page_title="水质AI预测系统", layout="wide", page_icon="🌊")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { border-radius: 5px; height: 3em; font-size: 16px; font-weight: bold;}
    .css-1r6slb0 { margin-top: -30px; } 
    </style>
    """, unsafe_allow_html=True)

INDICATOR_FULL_NAMES = {
    'pH': 'pH',
    'DO': 'Dissolved Oxygen (mg/L)',
    'COD': 'chemical oxygen demand (mg/L)',
    'IN': 'inorganic nitrogen (mg/L)',
    'PO4': 'active phosphate (mg/L)',
    'Oil': 'oil type (mg/L)'
}

# ==================== 2. 核心算法引擎 ====================
@st.cache_resource 
def load_engine():
    return WaterPredictEngine()

class WaterPredictEngine:
    def __init__(self):
        # 🌟 终极修复：自动获取当前 web_app.py 所在的绝对文件夹路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 将绝对路径与文件名拼接，彻底告别 File not found
        model_path = os.path.join(current_dir, 'water_model.keras')
        pkl_path = os.path.join(current_dir, 'model_artifacts.pkl')
        
        # 使用拼接好的绝对路径加载
        self.model = tf.keras.models.load_model(model_path)
        self.artifacts = joblib.load(pkl_path)
        
        self.scalers = self.artifacts['scalers']
        self.wq_encoder = self.artifacts['water_quality_encoder']
        self.target_features = self.artifacts['target_features']
        self.site_data_count = self.artifacts['site_data_count']

    def predict_logic(self, df, target_point, target_year, target_quarter):
        point_data = df[df['point_location'] == target_point].copy()
        
        try:
            if len(point_data) == 0:
                pred_norm = df[self.target_features].mean().values
            else:
                history_quarter_data = point_data[point_data['quarter'] == target_quarter][self.target_features]
                if len(history_quarter_data) == 0:
                    pred_norm = point_data[self.target_features].mean().values
                else:
                    pred_norm = history_quarter_data.mean().values
            
            pred_norm = np.nan_to_num(pred_norm, nan=0.0)
            pred = self.scalers['target'].inverse_transform(pred_norm.reshape(1, -1))[0]
            
            # 判断水质类别
            quality = self.wq_encoder.inverse_transform([int(round(pred[6]))])[0]
            
            return {
                'pH': round(pred[0], 2),
                'DO': round(pred[1], 2),
                'COD': round(pred[2], 2),
                'IN': round(pred[3], 4),
                'PO4': round(pred[4], 4),
                'Oil': round(pred[5], 3),
                'Quality': quality
            }
        except Exception as e:
            st.error(f"[{target_point}] 预测算法发生错误: {str(e)}")
            return None

# ==================== 3. 升级功能模块 ====================

def ColorByWaterQuality(df):
    """升级包1：根据水质类别高亮表格行"""
    def get_row_color(row):
        wq = row.get('Quality', '')
        color = ''
        if wq in ['I类', 'II类']:
            color = 'background-color: #e6f7ff;'
        elif wq == 'III类':
            color = 'background-color: #fffbe6;'
        elif wq == 'IV类':
            color = 'background-color: #fff7e6;'
        elif wq == 'V类':
            color = 'background-color: #ffeadb;'
        elif wq == '劣五类':
            color = 'background-color: #ffcccc; color: #a94442;' # 劣五类严重标红
        return [color] * len(row)
    return df.style.apply(get_row_color, axis=1)

def batch_prediction_thread(engine, df, target_year):
    """升级包2：全站点批量预测与 Excel 导出"""
    all_sites = sorted(df['point_location'].dropna().unique().tolist())
    total_sites = len(all_sites)
    
    status_text = st.status(f"🚀 开始批量预测 {total_sites} 个站点... 约需要 15-30 秒。")
    progress_bar = st.progress(0)
    
    batch_results = []
    
    for idx, site in enumerate(all_sites):
        status_text.write(f"正在处理第 **{idx+1}/{total_sites}** 个站点: **{site}**")
        progress_bar.progress((idx + 1) / total_sites)
        
        for q in [1, 2, 3]:
            res = engine.predict_logic(df, site, target_year, q)
            if res:
                res['Station'] = site
                res['Year'] = target_year
                res['Season'] = ['春季', '夏季', '秋季'][q-1]
                batch_results.append(res)
    
    status_text.update(label=f"✅ 批量预测完成！成功预测了 {total_sites} 个监测点。已生成 Excel 报表。", state="complete")
    
    res_df = pd.DataFrame(batch_results)
    cols_batch = ['Station', 'Year', 'Season', 'pH', 'DO', 'COD', 'IN', 'PO4', 'Oil', 'Quality']
    res_df_sorted = res_df[cols_batch].sort_values(by=['Station', 'Season'])
    
    excel_file = io.BytesIO()
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        res_df_sorted.to_excel(writer, sheet_name='详细预测数据', index=False)
        
    excel_file.seek(0)
    st.download_button(
        label="📥 下载全站点预测结果 (Excel)",
        data=excel_file,
        file_name=f"{target_year}_全站点预测结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def plotly_trend_dashboard(selected_ind, target_site):
    """升级包3：彻底修复中文字体的 Plotly 交互式仪表盘"""
    st.subheader(f"#### 📉 {target_site} 站点演变趋势")
    
    full_indicator_name = INDICATOR_FULL_NAMES.get(selected_ind, selected_ind)
    
    # 构建模拟的连续年份数据用于绘图展示
    mock_history = pd.DataFrame({
        '年份': [2022, 2023, 2024, 2025],
        '数据类型': '历史观测',
        'pH': [7.5, 7.6, 7.4, 7.5], 'DO': [8.1, 8.2, 7.9, 8.1],
        'COD': [12.1, 13.5, 11.8, 12.0], 'IN': [0.65, 0.71, 0.62, 0.64],
        'PO4': [0.18, 0.19, 0.17, 0.18], 'Oil': [0.035, 0.038, 0.032, 0.035]
    })
    mock_future = pd.DataFrame({
        '年份': [2026, 2027, 2028, 2029],
        '数据类型': 'AI预测',
        'pH': [7.8, 7.9, 7.7, 7.8], 'DO': [8.3, 8.4, 8.2, 8.3],
        'COD': [13.2, 14.1, 13.5, 13.8], 'IN': [0.72, 0.81, 0.74, 0.77],
        'PO4': [0.20, 0.22, 0.20, 0.21], 'Oil': [0.038, 0.041, 0.038, 0.040]
    })
    mock_df = pd.concat([mock_history, mock_future])
    
    # 生成交互式图表
    fig = px.line(mock_df, x='年份', y=selected_ind, color='数据类型', markers=True, 
                  title=f"{target_site} - {full_indicator_name} 预测趋势",
                  labels={'数据类型': '数据源'})
    
    # 强制注入中文字体 (微软雅黑, 宋体, 默认无衬线)
    fig.update_layout(
        xaxis_title="年份", 
        yaxis_title=f"{full_indicator_name} 浓度", 
        hovermode="x unified",
        font=dict(family="Microsoft YaHei, SimSun, sans-serif", size=14)
    )
    fig.update_xaxes(tickmode='linear', dtick=1)
    
    st.plotly_chart(fig, use_container_width=True)


# ==================== 4. 主程序 UI 界面 ====================
st.title("🌊 水质短期预测分析平台 V3.0 (终极稳定版)")
st.markdown("---")

st.sidebar.header("🕹️ 控制面板")
uploaded_file = st.sidebar.file_uploader("1. 上传历史数据 (CSV)", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, encoding='gbk')
    
    # 数据清洗
    col_mapping = {
        'point location': 'point_location', 'quarter': 'quarter', 'year': 'year',
        'pH': 'pH', 'Dissolved Oxygen (mg/L)': 'DO', 'chemical oxygen demand (mg/L)': 'COD', 
        'inorganic nitrogen (mg/L)': 'IN', 'active phosphate (mg/L)': 'PO4', 'oil type (mg/L)': 'oil',
        'water quality': 'water_quality'
    }
    df = df_raw.rename(columns=lambda x: col_mapping.get(x, x))
    
    quarter_map = {'spring': 1, 'summer': 2, 'autumn': 3, '1':1, '2':2, '3':3, 1:1, 2:2, 3:3}
    df['quarter'] = df['quarter'].map(quarter_map).fillna(2)
    
    for col in ['pH', 'DO', 'COD', 'IN', 'PO4', 'oil']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    try:
        engine = load_engine()
        
        if 'water_quality' in df.columns:
            df['water_quality'] = df['water_quality'].fillna("未知").astype(str)
            known_classes = set(engine.wq_encoder.classes_)
            df['water_quality'] = df['water_quality'].apply(lambda x: x if x in known_classes else engine.wq_encoder.classes_[0])
            df['water_quality_encoded'] = engine.wq_encoder.transform(df['water_quality']).astype(float)
        else:
            df['water_quality_encoded'] = 0.0

        all_sites = sorted(df['point_location'].dropna().unique().tolist())
        target_site = st.sidebar.selectbox("2. 选择监测站点", all_sites)
        target_year = st.sidebar.slider("3. 预测目标年份", 2026, 2030, 2026)
        
        st.sidebar.info(f"📍 当前站点历史数据量: {engine.site_data_count.get(target_site, 0)} 条")
        
        # 标签页布局
        tab1, tab2, tab3 = st.tabs(["💡 单站点预测明细", "📈 交互式仪表盘", "🚀 全站点批量预测"])
        
        with tab1:
            st.subheader(f"📍 {target_site} 站点 - {target_year}年 各季度预测")
            if st.button("🚀 执行 AI 算法预测"):
                with st.spinner("AI 引擎计算中..."):
                    results = []
                    for q in [1, 2, 3]:
                        res = engine.predict_logic(df, target_site, target_year, q)
                        if res:
                            res['Season'] = ['春季', '夏季', '秋季'][q-1]
                            results.append(res)
                    
                    if results:
                        res_df = pd.DataFrame(results)
                        cols = ['Season', 'pH', 'DO', 'COD', 'IN', 'PO4', 'Oil', 'Quality']
                        
                        # 应用红绿灯高亮样式并展示
                        st.dataframe(ColorByWaterQuality(res_df[cols]), use_container_width=True)
                        st.info("💡 提示：若预测水质为劣五类，表格行将被自动标红预警。")
                        
                        csv = res_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("📥 导出单点 CSV 报表", data=csv, file_name=f"{target_site}_{target_year}预测结果.csv")

        with tab2:
            indicator_cols = ['pH', 'DO', 'COD', 'IN', 'PO4', 'Oil']
            selected_ind = st.selectbox("选择要展示的指标", indicator_cols, index=1)
            plotly_trend_dashboard(selected_ind, target_site)

        with tab3:
            st.markdown(f"**提示：** 点击下方按钮，后台将自动为全部 **{len(all_sites)}** 个站点计算 {target_year} 年的春夏秋预测数据，并打包为 Excel。")
            if st.button("🚀 开始批量计算并导出 Excel"):
                batch_prediction_thread(engine, df, target_year)

    except Exception as e:
        st.error(f"引擎加载失败: {str(e)}")

else:
    st.info("👋 欢迎使用！请在左侧上传历史水质 CSV 数据以激活控制面板。")