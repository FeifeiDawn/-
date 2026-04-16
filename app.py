import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 全局配置与基础参数
# ==========================================
st.set_page_config(page_title="T0 SKU MRP仿真系统", layout="wide")
st.title("📦 供应链 MRP 自适应寻优沙盘 (统一量纲版)")

TOTAL_WEEKS = 52
BASE_DEMAND = 30
BASE_LT = 8
REVIEW_PERIOD = 2

# ==========================================
# 纯函数计算内核
# ==========================================
def get_forecast(mode_str, param_val, actual_arr, static_val):
    """基于过去的实际销量，生成每期最新的单点预测值"""
    arr = np.zeros(TOTAL_WEEKS + 1)
    arr[1] = static_val if "4." in mode_str else BASE_DEMAND
    for t in range(2, TOTAL_WEEKS + 1):
        if "1." in mode_str:
            n = int(param_val)
            arr[t] = BASE_DEMAND if t <= n else np.mean(actual_arr[t-n:t])
        elif "2." in mode_str:
            beta = float(param_val)
            arr[t] = beta * actual_arr[t-1] + (1 - beta) * arr[t-1]
        elif "3." in mode_str:
            arr[t] = actual_arr[t-1]
        elif "4." in mode_str:
            arr[t] = static_val
    return arr

def run_core_simulation(algo_p, air_p, actual_d, fore_d, eval_base_opt, order_base_opt, order_static_val, review_p, init_inv, lt_array, ss_w, lb_w, ub_w):
    history = []
    inventory = init_inv 
    pipeline = [{'eta': w, 'qty': BASE_DEMAND * review_p, 'type': 'SEA'} for w in range(1, BASE_LT + 1, review_p)]

    for t in range(1, TOTAL_WEEKS + 1):
        theoretical_demand = actual_d[t] # 理论需求
        current_forecast = fore_d[t]     # 站在第t周视角的最新预测
        current_lt = lt_array[t]

        # =========================================================
        # 核心架构升级：滚动预测投影 (Rolling Forecast Projection)
        # =========================================================
        proj_future_demand = current_forecast 

        # 1. 设定执行基准 (Order Base)
        if "静态基准" in order_base_opt:
            current_order_base = order_static_val
        else:
            current_order_base = proj_future_demand 
            
        # 2. 设定评价基准 (Eval Base)
        if "当周实际" in eval_base_opt:
            current_eval_base = (actual_d[t] + proj_future_demand * 3) / 4.0
        else:
            end_t = min(t + 4, TOTAL_WEEKS + 1) 
            current_eval_base = np.mean(actual_d[t:end_t]) # 事后上帝视角
        current_eval_base = max(current_eval_base, 1) # 兜底防除零

        # =========================================================
        
        # 动态水位计算内核
        obs_n = algo_p.get('obs_window', 4)
        lookback = min(t - 1, obs_n)
        if lookback >= 2:
            past_d = actual_d[t-lookback:t]
            past_lt = lt_array[t-lookback:t]
            d_avg, sigma_d = np.mean(past_d), np.std(past_d)
            l_avg, sigma_l = np.mean(past_lt), np.std(past_lt)
        else:
            d_avg, sigma_d = BASE_DEMAND, 0
            l_avg, sigma_l = BASE_LT, 0

        sigma_dl = np.sqrt(l_avg * (sigma_d**2) + (d_avg**2) * (sigma_l**2))
        z_factor = algo_p.get('z_factor', 1.0)
        # 动态目标 = 基础底座(独立安全库存设定+LT) * 预测基准 + Z * 联合标准差
        dynamic_target_pieces = (ss_w + BASE_LT) * current_order_base + z_factor * sigma_dl

        # --- 物流到港与入库 ---
        arrived_qty = sum(item['qty'] for item in pipeline if item['eta'] <= t)
        pipeline = [item for item in pipeline if item['eta'] > t]
        inventory += arrived_qty

        # --- 空运救火机制守护进程 (双限逻辑升级) ---
        air_qty_pieces = 0
        air_penalty = 0
        if air_p['enabled'] and t < TOTAL_WEEKS:
            expected_inv_next_week = inventory - current_forecast
            
            for item in pipeline:
                if item['eta'] == t + 1 and item['type'] == 'SEA':
                    expected_inv_next_week += item['qty']
            
            # 触发线 (跌破此线则报警)
            trigger_target = air_p['trigger_wos'] * proj_future_demand
            
            if expected_inv_next_week < trigger_target:
                # 补齐线 (一旦报警，直接补足到更安全的倍数)
                replenish_target = air_p['replenish_wos'] * proj_future_demand
                
                air_qty_pieces = max(replenish_target - expected_inv_next_week, air_p['moq'])
                pipeline.append({'eta': t + 1, 'qty': air_qty_pieces, 'type': 'AIR'})
                air_penalty = air_qty_pieces * air_p['unit_penalty'] # 空运罚分

        # --- 实际消耗与断货判定 ---
        actual_sales = min(inventory, theoretical_demand)
        inventory -= actual_sales
        lost_sales = theoretical_demand - actual_sales
        is_stockout = lost_sales > 0

        # --- 统一量纲(件数)损益评价 (计算得分) ---
        wos = inventory / current_eval_base if current_eval_base > 0 else 999
        
        # 转化为物理件数的上下界
        lower_bound_pieces = lb_w * current_eval_base
        upper_bound_pieces = ub_w * current_eval_base
        
        # 1. 断货丢失销量惩罚：每少1件扣5分
        stockout_penalty = lost_sales * 5.0
        
        # 2. 危险缺口惩罚：每低于下界1件扣0.3分
        understock_penalty = 0.0
        if inventory < lower_bound_pieces:
            understock_penalty = (lower_bound_pieces - inventory) * 0.3
            
        # 3. 压货惩罚：每多压1件扣0.1分
        overstock_penalty = 0.0
        if inventory > upper_bound_pieces:
            overstock_penalty = (inventory - upper_bound_pieces) * 0.1
            
        # 总分为各项惩罚成本的叠加 (包含本期触发的空运成本)
        score = air_penalty + stockout_penalty + understock_penalty + overstock_penalty

        # --- 海运发货执行 (全自动 AI 策略) ---
        sea_qty_pieces = 0
        current_pipeline_qty = sum(item['qty'] for item in pipeline)

        if t % review_p == 0:
            if theoretical_demand < current_forecast * algo_p['delay_trigger']:
                sea_qty_pieces = 0 
            else:
                gap = dynamic_target_pieces - (inventory + current_pipeline_qty)
                planned_order = (current_forecast * review_p) + (gap * algo_p['alpha'])
                moq_pieces = algo_p['moq_w'] * current_order_base
                if planned_order > 0:
                    sea_qty_pieces = max(planned_order, moq_pieces)

        if sea_qty_pieces > 0:
            pipeline.append({'eta': t + current_lt, 'qty': sea_qty_pieces, 'type': 'SEA'})

        history.append({
            'Week': t, 'Theoretical_Demand': theoretical_demand, 'Actual_Sales': actual_sales,
            'Lost_Sales': lost_sales, 'Is_Stockout': is_stockout, 'Forecast': current_forecast,
            'Eval_Base': current_eval_base, 'Inventory': inventory,
            'Pipeline_Qty': current_pipeline_qty, 'Order_Qty': sea_qty_pieces, 'Air_Qty': air_qty_pieces,
            'LT': current_lt, 'Order_ETA': t + current_lt if sea_qty_pieces > 0 else None,
            'Dynamic_Target': dynamic_target_pieces, 'Sigma_D': sigma_d, 'Sigma_L': sigma_l, 'Sigma_DL': sigma_dl,
            'WOS': wos, 'Score': score, 
            'Penalty_Stockout': stockout_penalty, 'Penalty_Under': understock_penalty, 
            'Penalty_Over': overstock_penalty, 'Penalty_Air': air_penalty
        })
        
    df = pd.DataFrame(history)
    return df, df['Score'].sum()

# ==========================================
# 左侧边栏：环境与约束设定
# ==========================================
sidebar = st.sidebar
sidebar.header("🛠️ 1. 业务环境情景构建器")

sidebar.subheader("A. 销量扰动 (真实市场)")
enable_crisis_d = sidebar.checkbox("📉 启用局部销量突变", value=True)
crisis_week_d = sidebar.slider("销量突变：第 X 周起发生", 1, 52, 15, disabled=not enable_crisis_d)
crisis_pct_change = sidebar.slider("销量突变幅度 (%)", -90, 200, 200, step=10, disabled=not enable_crisis_d) / 100.0
demand_noise = sidebar.slider("日常销量白噪音波动 (%)", 0, 50, 15, step=5) / 100.0

sidebar.subheader("B. 物流随机波动 (真实供应)")
lt_min, lt_max = sidebar.slider("海运周期三角分布区间 (周)", min_value=4, max_value=16, value=(6, 12))

sidebar.subheader("C. 既定预测黑盒")
forecast_mode_ui = sidebar.radio("系统内部预测逻辑", ["1. 简单移动平均法", "2. 指数平滑法", "3. 最新实际销量", "4. 静态固定"], index=0)
ma_n_ui = sidebar.slider("SMA观测期 (周)", 1, 12, 4) if "1." in forecast_mode_ui else 4
ema_beta_ui = sidebar.slider("EMA平滑系数 β", 0.0, 1.0, 0.3) if "2." in forecast_mode_ui else 0.3
static_forecast_val_ui = sidebar.number_input("固定锚定预测值", min_value=1, value=BASE_DEMAND) if "4." in forecast_mode_ui else BASE_DEMAND

sidebar.subheader("D. 初始与硬约束")
initial_inventory = sidebar.number_input("第 0 周初始在库件数", value=BASE_DEMAND * 4, step=10)
moq_w_ui = sidebar.number_input("海运 MOQ 限制 (周)", value=0.0, step=0.5)

sidebar.subheader("E. 紧急救火机制 (空运/快船)")
st.sidebar.info("预判下周不足时强制空运。注意：由于启用了 AI 联合寻优，触发阈值和补齐目标将被 AI 接管探索，您只需设定基础运费惩罚即可。")
enable_air = sidebar.checkbox("✈️ 启用确定性空运救火机制", value=False)
air_moq = sidebar.number_input("空运起订量 (件)", value=20, step=10, disabled=not enable_air)
air_unit_penalty = sidebar.number_input("单件空运罚分", value=2.0, step=0.5, format="%.2f", disabled=not enable_air)
# 界面传参将用于非优化场景（如果还有的话）或占位
air_params = {'enabled': enable_air, 'trigger_wos': 1.5, 'replenish_wos': 3.0, 'moq': air_moq, 'unit_penalty': air_unit_penalty}

# 预计算全局变量
np.random.seed(42)
actual_demand_arr = np.zeros(TOTAL_WEEKS + 1)
for t in range(1, TOTAL_WEEKS + 1):
    base_t = BASE_DEMAND * (1 + crisis_pct_change) if enable_crisis_d and t >= crisis_week_d else BASE_DEMAND
    actual_demand_arr[t] = max(0, base_t * np.random.normal(1.0, demand_noise)) 

lt_arr = np.round(np.random.triangular(lt_min, BASE_LT, lt_max, TOTAL_WEEKS + 1)).astype(int)
lt_arr = np.maximum(lt_arr, 1)

# ==========================================
# 主界面 1：基准设定与 KPI 规则
# ==========================================
st.markdown("### 📊 评价体系与执行基准设定")
col_eval, col_order, col_ss, col_lb, col_ub = st.columns([1.5, 1.5, 1, 1, 1])
with col_eval: 
    eval_base_opt = st.selectbox("1. KPI 评价与红线基准：", ["当周实际销量与未来3周预测均值 (混合动态视角)", "本周及未来 3 周的实际销量均值 (上帝客观视角)"], index=0)
with col_order: 
    order_base_opt = st.selectbox("2. MRP 执行计算基准：", ["动态推演：ETA及未来3周的平均预期", "静态基准：固定数值"], index=0)
    order_static_val = st.number_input("静态执行基准值 (件)", min_value=1, value=30) if "静态基准" in order_base_opt else BASE_DEMAND
with col_ss: ss_w = st.number_input("🛡️ 目标安全库存(周)", value=2.0, step=0.5, help="独立于打分系统，用于计算发货水位的物理底座")
with col_lb: lower_bound_w = st.number_input("⬇️ 评分下界(缺口扣分)", value=2.0, step=0.5)
with col_ub: upper_bound_w = st.number_input("⬆️ 评分上界(压货扣分)", value=5.0, step=0.5)

current_env_state = (
    enable_crisis_d, crisis_week_d, crisis_pct_change, demand_noise, lt_min, lt_max, 
    forecast_mode_ui, ma_n_ui, ema_beta_ui, static_forecast_val_ui, initial_inventory, moq_w_ui, 
    enable_air, air_moq, air_unit_penalty,
    eval_base_opt, order_base_opt, order_static_val, ss_w, lower_bound_w, upper_bound_w
)
if 'env_state' not in st.session_state or st.session_state['env_state'] != current_env_state:
    st.session_state.pop('ai_res', None)
    st.session_state['env_state'] = current_env_state

forecast_arr_ui = get_forecast(forecast_mode_ui, ma_n_ui if "1." in forecast_mode_ui else ema_beta_ui, actual_demand_arr, static_forecast_val_ui)

# ==========================================
# 主界面 2：AI 自动寻优沙盘
# ==========================================
st.markdown("---")
st.header("🚀 AI 海空联运联合寻优中心 (Sea-Air Joint Optimization)")
st.info("系统将在设定的环境下执行**数百次**沙盘推演，并自动输出总成本最低的【海空联运最优配置】。在现代 CPU 下，这仅需不到 0.5 秒。")

with st.expander("💡 了解 AI 海空联运决策推演逻辑"):
    st.markdown("""
    1. **设定不变环境**：系统将左侧市场波动、物流阻碍以及预测模型等作为不可更改的外部环境固定下来。
    2. **遍历海运策略空间**：网格搜索发货频率 `RP`、安全系数 `Z`、缺口平滑系数 `α`、停发熔断线。
    3. **遍历空运救火空间**：如果开启了空运，AI 将额外搜索最佳的**空运触发线**（1.0周, 1.5周, 2.0周等）和**空运补齐目标**（2.0周, 3.0周, 4.0周）。
    4. **最低罚分裁决**：对比每次推演产生的总惩罚分（断货罚分 + 压货罚分 + 空运救火单件成本），找出全局最低损益的商业平衡解。
    """)

if st.button("🌟 启动 AI 海空联合沙盘推演", type="primary"):
    with st.spinner("AI 网络搜索引擎高负荷运算中..."):
        # 海运网格
        grid_rp = [1, 2]
        grid_alpha = [0.3, 0.6, 1.0]
        grid_z = [0.5, 1.0, 1.5, 2.0]
        grid_trigger = [0.6, 0.8]
        
        # 空运网格：动态接管
        if air_params['enabled']:
            grid_air_t = [1.0, 1.5, 2.0]        # 触发阈值选项
            grid_air_r = [2.0, 3.0, 4.0]        # 补齐目标选项
        else:
            grid_air_t = [1.5] # 占位
            grid_air_r = [3.0] # 占位
            
        best_score, best_avg_inv, best_params, best_df = float('inf'), float('inf'), {}, None
        
        fore_arr = forecast_arr_ui
        
        for rp in grid_rp:
            for a in grid_alpha:
                for z in grid_z:
                    for trig in grid_trigger:
                        for a_t in grid_air_t:
                            for a_r in grid_air_r:
                                # 排除不合理的逻辑：补齐目标必须大于触发线
                                if air_params['enabled'] and a_r <= a_t:
                                    continue
                                
                                # 将 AI 寻找到的空运参数装载
                                current_air_p = air_params.copy()
                                current_air_p['trigger_wos'] = a_t
                                current_air_p['replenish_wos'] = a_r
                                
                                a_p = {'z_factor': z, 'alpha': a, 'moq_w': moq_w_ui, 'obs_window': 4, 'delay_trigger': trig}
                                df_t, s_t = run_core_simulation(
                                    a_p, current_air_p, actual_demand_arr, fore_arr, 
                                    eval_base_opt, order_base_opt, order_static_val, 
                                    rp, initial_inventory, lt_arr, ss_w, lower_bound_w, upper_bound_w
                                )
                                
                                inv_avg = df_t['Inventory'].mean()
                                if s_t < best_score or (s_t == best_score and inv_avg < best_avg_inv):
                                    best_score = s_t
                                    best_avg_inv = inv_avg
                                    best_params = {'rp':rp, 'alpha':a, 'z':z, 'trig':trig, 'air_t': a_t, 'air_r': a_r}
                                    best_df = df_t
                                    
        st.session_state['ai_res'] = {'params': best_params, 'df': best_df, 'score': best_score}

if 'ai_res' in st.session_state:
    df_res, total_score, bp = st.session_state['ai_res']['df'], st.session_state['ai_res']['score'], st.session_state['ai_res']['params']
    msg = f"**✅ 联合寻优完成！最优海运应对策略 (总罚分 {total_score:.1f})**: 每 `{bp['rp']}` 周 1 发 | 平滑 `α={bp['alpha']}` | 最佳海运安全系数 `Z={bp['z']}`"
    if air_params['enabled']:
        msg += f"\n\n**✈️ 最优空运救火策略**: 预期跌破 `{bp['air_t']}`周 时触发报警 | 触发后一次性空运补足至 `{bp['air_r']}`周"
    st.success(msg)
else: 
    st.warning("👈 请调整左侧环境参数后，点击上方按钮开始生成推演结果")
    st.stop()

# ==========================================
# 结果视图展示区
# ==========================================
st.markdown("---")
st.header("⏳ 结果视图：最优策略下的时空表现")

col_score, col_popover, _, _ = st.columns([1.5, 2, 1, 1])
with col_score: st.markdown(f"<div style='font-size:24px; font-weight:bold; margin-top:5px;'>🎯 总分: <span style='color:#E63946;'>{total_score:.1f} 分</span></div>", unsafe_allow_html=True)
with col_popover:
    with st.popover("📜 查看当前评分规则 (按物理件数核算)"):
        st.markdown(f"""
        🟢 **健康水位**：在库 `{lower_bound_w}`~`{upper_bound_w}` 周 (0分)
        
        🟠 **危险缺口**：每低于下界 1 件，扣 `0.3` 分/周
        
        🔴 **销量丢失**：发生断货时，每丢失 1 件销量，扣 `5.0` 分
        
        🟡 **压货惩罚**：每高于上界 1 件，扣 `0.1` 分/周
        
        ✈️ **空运成本**：每发 1 件空运，扣 `{air_params['unit_penalty']}` 分
        """)

cols = st.columns(5)
cols[0].metric("丢失销量总计", f"{int(df_res['Lost_Sales'].sum())} 件")
cols[1].metric("空运件数总计", f"{int(df_res['Air_Qty'].sum())} 件")
cols[2].metric("严重断货发生周数", f"{len(df_res[df_res['Is_Stockout']])} 周")
cols[3].metric("压货扣分期", f"{len(df_res[df_res['Penalty_Over'] > 0])} 周")
cols[4].metric("免罚分健康期", f"{len(df_res[df_res['Score'] == 0])} 周")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("图表 A: 真实需求与发货应对 (红点代表发生断货遗失销量)", "图表 B: 绝对管线水位与评估红线"))

fig.update_layout(barmode='stack')

fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Actual_Sales'], name="实际销量 (受限于库存)", line=dict(color='#0077B6', width=2), hovertemplate="实际售出: %{y:.0f}件"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Theoretical_Demand'], name="理论需求 (真实验配)", line=dict(color='#0077B6', width=2, dash='dash'), hovertemplate="理论想卖: %{y:.0f}件"), row=1, col=1)

stockouts = df_res[df_res['Is_Stockout']]
if not stockouts.empty:
    fig.add_trace(go.Scatter(x=stockouts['Week'], y=stockouts['Theoretical_Demand'], mode='markers', marker=dict(color='red', size=8), name="断货损失点", customdata=stockouts['Lost_Sales'], hovertemplate="💔发生断货！<br>错失销量: %{customdata:.0f}件"), row=1, col=1)

fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Forecast'], name="系统滞后预测", line=dict(color='#D62828', dash='dot'), hovertemplate="预测: %{y:.0f}件"), row=1, col=1)

c_data_a = np.column_stack((df_res['Order_Qty']/df_res['Eval_Base'], df_res.apply(lambda r: f"第 {int(r['Order_ETA'])} 周" if pd.notnull(r['Order_ETA']) else "", axis=1)))
fig.add_trace(go.Bar(x=df_res['Week'], y=df_res['Order_Qty'], name="海运实发", marker_color='#F77F00', customdata=c_data_a, hovertemplate="海运: %{y:.0f}件<br>预计到港: <b>%{customdata[1]}</b>"), row=1, col=1)

if air_params['enabled']:
    fig.add_trace(go.Bar(x=df_res['Week'], y=df_res['Air_Qty'], name="✈️ 空运救火", marker_color='#D62828', hovertemplate="空运紧急补发: %{y:.0f}件"), row=1, col=1)

fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Inventory'], name="在库数量", fill='tozeroy', marker_color='#2A9D8F', customdata=df_res['WOS'], hovertemplate="在库: %{y:.0f}件 (%{customdata:.1f}周)"), row=2, col=1)
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Inventory'] + df_res['Pipeline_Qty'], name="总管线(在途+在库)", fill='tonexty', marker_color='#A8DADC', hovertemplate="管线总计: %{y:.0f}件"), row=2, col=1)

c_data_b = np.column_stack((df_res['Sigma_D'], df_res['Sigma_L'], df_res['Sigma_DL'], df_res['Dynamic_Target']/df_res['Eval_Base']))
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Dynamic_Target'], name="自适应目标水位 (联合风险驱动)", line=dict(color="purple", dash='dot'), customdata=c_data_b, hovertemplate="自适应水位: %{y:.0f}件 (%{customdata[3]:.1f}周)<br>--- 内核透视 ---<br>销量波动(σD): %{customdata[0]:.1f}件<br>物流波动(σL): %{customdata[1]:.1f}周<br>联合风险缺口: %{customdata[2]:.1f}件"), row=2, col=1)

fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Eval_Base'] * ss_w, name=f"底线基座 (在库{ss_w}周)", line=dict(color="#E63946", width=1, dash='solid'), mode='lines', hoverinfo='skip'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Eval_Base'] * lower_bound_w, name=f"打分下界 ({lower_bound_w}周)", line=dict(color="blue", width=1, dash='dash'), mode='lines', hoverinfo='skip'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_res['Week'], y=df_res['Eval_Base'] * upper_bound_w, name=f"打分上界 ({upper_bound_w}周)", line=dict(color="blue", width=1, dash='dash'), mode='lines', hoverinfo='skip'), row=2, col=1)

fig.update_layout(height=800, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)
