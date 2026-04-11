import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 全局配置与基础参数
# ==========================================
st.set_page_config(page_title="T0 SKU MRP仿真系统", layout="wide")
st.title("📦 供应链 MRP 深度仿真与归因沙盘 (周数驱动版)")

TOTAL_WEEKS = 52
BASE_DEMAND = 30
BASE_LT = 8
REVIEW_PERIOD = 2

# ==========================================
# 提前预计算：需求与预测阵列 (解耦件数与周数的核心)
# ==========================================
sidebar = st.sidebar
sidebar.header("🛠️ 1. 情景构建器 (危机注入)")

# -- 注入危机 --
sidebar.subheader("A. 销量情况")
enable_crisis_d = sidebar.checkbox("📉 启用销量突变", value=True)
crisis_week_d = sidebar.slider("销量突变：第 X 周开始发生", 1, 52, 15, disabled=not enable_crisis_d)
crisis_pct_change = sidebar.slider("销量变化幅度 (%)", min_value=-90, max_value=200, value=-50, step=10, help="-50%代表跌一半，+100%代表翻倍", disabled=not enable_crisis_d) / 100.0
demand_noise = sidebar.slider("日常销量随机波动 (%)", min_value=0, max_value=50, value=10, step=5, help="引入真实世界中每天/每周销量的随机白噪音") / 100.0

sidebar.subheader("B. 物流情况")
crisis_week_l = sidebar.slider("物流变化：第 Z 周开始发生", 1, 52, 20)
new_lt = sidebar.slider("变化后的海运前置期 (周)", min_value=4, max_value=15, value=8, help="默认8周，往右拖动代表港口拥堵/红海危机延误，往左代表加速")

# -- 预测模式 --
sidebar.subheader("C. 系统预测算法反应 (滚动更新)")
forecast_mode = sidebar.radio(
    "预测逻辑 (模拟 MRP 内部黑盒)",
    options=[
        "1. 简单移动平均法 (SMA, 过去 N 周均值)",
        "2. 指数平滑法 (EMA)",
        "3. 最新实际销量 (Naive, 上周实际)",
        "4. 锚定剧本法 (Static, 固定在特定值)"
    ],
    index=0
)
if "1." in forecast_mode:
    ma_n = sidebar.slider("N周 (迟钝算法参数)", 1, 12, 4)
elif "2." in forecast_mode:
    ema_beta = sidebar.slider("平滑系数 β (Beta)", 0.0, 1.0, 0.3, step=0.1, help="最新销量占预测的比重。越大越敏感，越小越平缓。")
elif "4." in forecast_mode:
    static_forecast_val = sidebar.number_input("固定锚定预测值", min_value=1, value=BASE_DEMAND)

# -> 预计算全年 52 周的实际销量与预测销量
np.random.seed(42) # 固定随机种子以保证调整其他参数时曲线不会乱跳
actual_demand_arr = np.zeros(TOTAL_WEEKS + 1)
for t in range(1, TOTAL_WEEKS + 1):
    if enable_crisis_d and t >= crisis_week_d:
        base_t = BASE_DEMAND * (1 + crisis_pct_change)
    else:
        base_t = BASE_DEMAND
    # 加入随机扰动
    noise_factor = np.random.normal(1.0, demand_noise)
    actual_demand_arr[t] = max(0, base_t * noise_factor) # 保证销量不为负数

forecast_arr = np.zeros(TOTAL_WEEKS + 1)
# 第一周缺乏历史数据，赋予初始值
forecast_arr[1] = static_forecast_val if "4." in forecast_mode else BASE_DEMAND

for t in range(2, TOTAL_WEEKS + 1):
    if "1." in forecast_mode:
        if t <= ma_n:
            forecast_arr[t] = BASE_DEMAND
        else:
            forecast_arr[t] = np.mean(actual_demand_arr[t-ma_n:t])
    elif "2." in forecast_mode:
        forecast_arr[t] = ema_beta * actual_demand_arr[t-1] + (1 - ema_beta) * forecast_arr[t-1]
    elif "3." in forecast_mode:
        forecast_arr[t] = actual_demand_arr[t-1]
    elif "4." in forecast_mode:
        forecast_arr[t] = static_forecast_val

# ==========================================
# 发货策略选择台 (基准与模式)
# ==========================================
st.header("⚙️ 2. 发货策略选择台 (周数调拨)")
st.markdown("所有的目标与发货指令将以**相对时间(周数)**为单位。请选择这些周数换算为实际发货件数时的**基准底数**：")

col_ref1, col_ref2 = st.columns([1, 1])
with col_ref1:
    base_ref_opt = st.selectbox(
        "选择【1周发货量】对应的件数基准：",
        ["动态基准：当周实际销量", "动态基准：当周系统预测销量", "超前预判：海运LT到货那周的预测销量", "静态基准：固定数值"],
        index=1
    )
with col_ref2:
    if "静态基准" in base_ref_opt:
        static_base_val = st.number_input("设定静态基准值 (件/周)", min_value=1, value=30)
    else:
        static_base_val = BASE_DEMAND
        st.write("") # 占位

# -> 计算动态的“周转件数转化基底”
base_qty_arr = np.zeros(TOTAL_WEEKS + 1)
for t in range(1, TOTAL_WEEKS + 1):
    if "当周实际" in base_ref_opt:
        base_qty_arr[t] = actual_demand_arr[t]
    elif "当周系统预测" in base_ref_opt:
        base_qty_arr[t] = forecast_arr[t]
    elif "海运LT" in base_ref_opt:
        current_lt = BASE_LT if t < crisis_week_l else new_lt
        arrival_week = min(t + current_lt, TOTAL_WEEKS)
        base_qty_arr[t] = forecast_arr[arrival_week]
    elif "静态基准" in base_ref_opt:
        base_qty_arr[t] = static_base_val
    
    # 防止作为除数的基准跌到极低导致发货量异常，设一个极小值兜底
    base_qty_arr[t] = max(base_qty_arr[t], 1)

st.markdown("---")
policy_mode = st.radio("选择供应链干预模式：", ["🔥 模式一：人工干预 (可视化时间轴调音台)", "🤖 模式二：算法对决 (自动寻找平滑发货点)"], horizontal=True)

manual_orders_weeks = np.zeros(TOTAL_WEEKS + 1)
algo_params = {}

if "模式一" in policy_mode:
    st.markdown("#### 🎛️ 时间轴发货调音台")
    st.info("👇 在下方横向数据格中输入您决定的**发货周数**，上方的柱状图会实时弹起，并在柱尖显示折算后的**实际发货件数**。")
    
    # 初始化参考发货周数
    init_orders_w = [REVIEW_PERIOD if w % REVIEW_PERIOD == 0 else 0 for w in range(1, TOTAL_WEEKS + 1)]
    
    # 横向时间轴输入器 (Data Editor)
    df_editor = pd.DataFrame([init_orders_w], columns=[str(i) for i in range(1, TOTAL_WEEKS + 1)], index=["填入发货量(周数)"])
    edited_df = st.data_editor(df_editor, use_container_width=True)
    
    # 提取周数并转换为件数
    manual_orders_pieces = np.zeros(TOTAL_WEEKS + 1)
    for i in range(1, TOTAL_WEEKS + 1):
        w_val = float(edited_df.iloc[0][str(i)])
        manual_orders_weeks[i] = w_val
        manual_orders_pieces[i] = w_val * base_qty_arr[i]
        
    # 渲染带有交互标签的实时柱形图
    fig_manual = go.Figure(data=[go.Bar(
        x=list(range(1, 53)), 
        y=manual_orders_weeks[1:], 
        text=[f"{p:.0f}件" if p>0 else "" for p in manual_orders_pieces[1:]],
        textposition='outside',
        marker_color='#FF9F1C',
        hovertemplate='第%{x}周发货<br>发货周数: %{y}周<br>实际发出: %{text}<extra></extra>'
    )])
    fig_manual.update_layout(
        height=280, 
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis_title="指令：发货(周)量"
    )
    st.plotly_chart(fig_manual, use_container_width=True)

else:
    st.markdown("#### ⚙️ 算法调优参数 (输入单位均为：周)")
    col1, col2, col3, col4 = st.columns(4)
    algo_params['target_level_w'] = col1.number_input("④目标管线总水位 (在库+在途周数)", value=12.0, step=1.0)
    algo_params['alpha'] = col2.slider("②发货平滑系数 (Alpha)", 0.0, 1.0, 0.5, help="计划发货量 = 预测 + (目标-实际)*Alpha")
    algo_params['moq_w'] = col3.number_input("③最低发货量限制 (MOQ周数)", value=0.0, step=0.5)
    algo_params['delay_trigger'] = col4.slider("①紧急熔断：销量<预测的X%时暂停发货", 0, 100, 70) / 100.0

st.markdown("---")

# ==========================================
# 核心仿真引擎跑批
# ==========================================
def run_simulation():
    history = []
    inventory = BASE_DEMAND * 4 # 初始健康在库 4周
    pipeline = [{'eta': w, 'qty': BASE_DEMAND * REVIEW_PERIOD} for w in range(1, BASE_LT + 1, REVIEW_PERIOD)]

    for t in range(1, TOTAL_WEEKS + 1):
        current_demand = actual_demand_arr[t]
        current_forecast = forecast_arr[t]
        current_base_qty = base_qty_arr[t] # 动态换算基准

        # --- 状态引擎：物流变化干扰 ---
        current_lt = BASE_LT
        if t == crisis_week_l:
            # 扰动发生周，所有海上的货也会受影响
            delay_diff = new_lt - BASE_LT
            for item in pipeline:
                item['eta'] += delay_diff
        if t >= crisis_week_l:
            current_lt = new_lt

        # --- 状态引擎：管线推进与入库 ---
        arrived_qty = 0
        remaining_pipeline = []
        for item in pipeline:
            if item['eta'] <= t:
                arrived_qty += item['qty']
            else:
                remaining_pipeline.append(item)
        pipeline = remaining_pipeline
        inventory += arrived_qty

        # --- 状态引擎：实际消耗 ---
        actual_sales = min(inventory, current_demand)
        inventory -= actual_sales

        # --- 评价引擎：WOS 计算与打分 ---
        wos = inventory / current_demand if current_demand > 0 else 99
        score = 0
        if wos < 2: score = 10
        elif 2 <= wos < 3: score = 3
        elif 3 <= wos <= 5: score = 0
        elif 5 < wos <= 8: score = 1
        elif wos > 8: score = 5

        # --- 策略引擎：计算发货 ---
        order_qty_pieces = 0
        current_pipeline_qty = sum(item['qty'] for item in pipeline)

        if "模式一" in policy_mode:
            # 模式1: 将人工定好的周数转化为件数
            order_qty_pieces = manual_orders_pieces[t]
        else:
            # 模式2: 算法动态计算 (内核为绝对件数推演)
            if t % REVIEW_PERIOD == 0:
                if current_demand < current_forecast * algo_params['delay_trigger']:
                    order_qty_pieces = 0 # 熔断
                else:
                    target_total_qty = algo_params['target_level_w'] * current_base_qty
                    gap = target_total_qty - (inventory + current_pipeline_qty)
                    # 计划发货件数
                    planned_order = (current_forecast * REVIEW_PERIOD) + (gap * algo_params['alpha'])
                    moq_pieces = algo_params['moq_w'] * current_base_qty
                    
                    if planned_order > 0:
                        order_qty_pieces = max(planned_order, moq_pieces)

        if order_qty_pieces > 0:
            pipeline.append({'eta': t + current_lt, 'qty': order_qty_pieces})

        history.append({
            'Week': t,
            'Demand': current_demand,
            'Forecast': current_forecast,
            'Inventory': inventory,
            'Pipeline_Qty': current_pipeline_qty,
            'Order_Qty': order_qty_pieces,
            'WOS': wos,
            'Score': score
        })
        
    return pd.DataFrame(history)

# ==========================================
# 时间机器输出结果
# ==========================================
st.header("⏳ 3. 时间机器跑批与结果视图")

df_res = run_simulation()
total_score = df_res['Score'].sum()

# 评分看板 (带悬浮规则说明)
col_score, col_popover = st.columns([1, 4])
with col_score:
    st.markdown(f"<div style='font-size:24px; font-weight:bold; margin-top:5px;'>🎯 总分: <span style='color:#E63946;'>{total_score} 分</span></div>", unsafe_allow_html=True)
with col_popover:
    with st.popover("📜 查看分数评定规则"):
        st.markdown("""
        **损益惩罚分数规则 (寻找最低分策略)：**
        * 🔴 **10分：极度断货惩罚** (在库 < 2 周) - 流量权重降级风险极大
        * 🔴 **5分：重度压货惩罚** (在库 > 8 周) - 资金占用巨大，仓储面临超龄
        * 🟠 **3分：断货预警状态** (2周 <= 在库 < 3 周)
        * 🟡 **1分：轻度压货惩罚** (5周 < 在库 <= 8 周)
        * 🟢 **0分：完美健康水位** (3周 <= 在库 <= 5 周)
        """)

st.write("") # padding
cols = st.columns(4)
cols[0].metric("极度断货次数 (10分)", len(df_res[df_res['WOS'] < 2]))
cols[1].metric("断货预警次数 (3分)", len(df_res[(df_res['WOS'] >= 2) & (df_res['WOS'] < 3)]))
cols[2].metric("重度压货次数 (5分)", len(df_res[df_res['WOS'] > 8]))
cols[3].metric("完美健康周数 (0分)", len(df_res[(df_res['WOS'] >= 3) & (df_res['WOS'] <= 5)]))

# 渲染图表
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("图表 A：供需对抗图 (所有显示均为件数)", "图表 B：全链路水位图 (在途+在库)")
)

# 图 A
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Demand'], name="实际销量", line=dict(color='#0077B6', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Forecast'], name="系统预测", line=dict(color='#D62828', width=2, dash='dot')), row=1, col=1)
fig.add_trace(go.Bar(x=df_res['Week'], y=df_res['Order_Qty'], name="实发件数", marker_color='#F77F00', opacity=0.7), row=1, col=1)

# 图 B
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Inventory'], name="在库数量", fill='tozeroy', marker_color='#2A9D8F'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Inventory'] + df_res['Pipeline_Qty'], name="总管线(在途+在库)", fill='tonexty', marker_color='#A8DADC'), row=2, col=1)

# 动态红线 (以初始基准件数画一条参考线)
fig.add_hline(y=BASE_DEMAND * 4, line_dash="solid", line_color="#E63946", annotation_text="静态断货底线 (在库4周)", row=2, col=1)
fig.add_hline(y=BASE_DEMAND * 12, line_dash="dash", line_color="#E63946", annotation_text="静态压货红线 (管线12周)", row=2, col=1)

fig.update_layout(height=700, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)