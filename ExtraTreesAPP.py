import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components

# 加载保存的 ExtraTrees 模型
model = joblib.load('ExtraTrees.pkl')

# 特征范围定义
feature_ranges = {
    "AGE": {"type": "numerical", "min": 0.0, "max": 18.0, "default": 5.0},
    "WT": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 20.0},
    "BUN": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0},
    "SCR": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 35.0},
    "CLCR": {"type": "numerical", "min": 0.0, "max": 500.0, "default": 90.0},
    "ALT": {"type": "numerical", "min": 0.0, "max": 500.0, "default": 20.0},
    "AST": {"type": "numerical", "min": 0.0, "max": 500.0, "default": 20.0},
    "TBIL": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0},
    "DBIL": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0},
    "HCT": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 35.0},
    "MCH": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 30.0},
    "Cmin": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 15.0}
}

# Streamlit 界面
st.title("Antiepileptic Drug (OXC) Treatment Outcome Prediction with SHAP Visualization")
st.write("""
This app predicts the likelihood of antiepileptic drug (OXC) treatment outcome based on input features.
Select the ET model, input feature values, and get predictions and probability estimates.
""")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 确保特征顺序一致
feature_order = model.feature_names_in_
features = pd.DataFrame(features, columns=feature_order)

# 确保输入数据为浮点数
features = features.astype(float)

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    st.write(f"Based on feature values, predicted possibility of good responder is {probability:.2f}%")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 生成 SHAP 力图
    class_index = predicted_class
    html_output = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[class_index],
        features,
        show=False
    )

    # 将 HTML 内容嵌入 Streamlit
    components.html(html_output.html(), height=200)
