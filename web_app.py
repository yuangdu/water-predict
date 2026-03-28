# web_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from backend_engine import WaterPredictor  # 🌟 核心：导入后厨引擎

st.set_page_config(page_title="水质 AI 预测分析平台", layout="wide", page_icon="🌊")

# 初始化后端引擎 (只在启动时加载一次模型)
@st.cache_resource
def load_engine():
    return WaterPredictor()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"引擎初始化失败，请确保模型文件(.keras 和 .pkl)存在。错误信息: {e}")
    st.stop()

st.title("🌊 水质短期预测分析平台 (全栈耦合版)")
st.markdown("---")

st.sidebar.header("🕹️ 控制面板")
st.sidebar.markdown("请上传 **2017-2025 常规6项原始数据 (CSV)**：")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])

if uploaded_file:
    # 1. 前台拿到原始食材
    raw_df = pd.read_csv(uploaded_file, encoding='gbk')
    
    st.sidebar.info("已读取原始数据，准备就绪。")
    
    # 点击按钮触发耦合流水线
    if st.button("🚀 启动深度学习滚动预测 (2026-2030)", type="primary"):
        with st.spinner("🧠 后台 AI 引擎正在疯狂运算中，请稍候..."):
            
            # 2. 核心对接：前台把食材扔给后厨，拿到做好的报表
            result_df = engine.process_and_predict_all(raw_df)
            
            # 将结果暂存入 session_state 以供页面切换使用
            st.session_state['pred_results'] = result_df
            st.success("✅ 全网点批量预测完成！")

if 'pred_results' in st.session_state:
    df_pred = st.session_state['pred_results']
    
    all_sites = sorted(df_pred['站点'].unique().tolist())
    target_site = st.selectbox("🎯 选择要查看预测明细的监测站点", all_sites)
    
    site_data = df_pred[df_pred['站点'] == target_site].copy()
    
    tab1, tab2 = st.tabs(["💡 AI 预测报表", "📈 演变趋势交互图"])
    
    with tab1:
        st.subheader(f"📍 {target_site} 站点 - (2026-2030) 预测结果")
        display_cols = ['年份', '季度名称', 'pH', 'DO', 'COD', 'IN', 'PO4', 'oil', 'water_quality', '警告信息']
        st.dataframe(site_data[display_cols], use_container_width=True)

    with tab2:
        st.subheader(f"📉 {target_site} 站点 - 理化指标演变趋势")
        indicators = ['pH', 'DO', 'COD', 'IN', 'PO4', 'oil']
        selected_col = st.selectbox("选择要展示的指标", indicators)
        
        site_data['预测日期'] = pd.to_datetime(site_data['预测日期'])
        site_data = site_data.sort_values('预测日期')
        
        fig = px.line(site_data, x='预测日期', y=selected_col, markers=True)
        
        fig.update_layout(
            xaxis_title="预测时间", 
            yaxis_title=selected_col, 
            hovermode="x unified"
        )
        fig.update_xaxes(autorange="reversed")
        
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 请在左侧上传原始水质数据，并点击预测按钮启动系统。")