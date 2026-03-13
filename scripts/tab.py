import pandas as pd

# 创建数据
data = {
    "方法": ["FedAvg", "IMS", "FedUp", "NOT", "Ours", "A4FL"],
    "BadNets": [13289, 56, 101, 158, 178, 24229],
    "DBA": [13959, 87, 97, 212, 202, 24259],
    "Neurotoxin": [12613, 91, 97, 210, 189, 24322],
    "PGD": [13964, 95, 110, 210, 192, 23570],
    "额外平均时间": [0.00, 82.25, 101.25, 197.50, 190.25, None],
    "总时间": [
        "13456.25（基线）",
        "13538.50（+0.61%）",
        "13557.50（+0.75%）",
        "13653.75（+1.47%）",
        "13646.50（+1.41%，比NOT少7.25）",
        "24095.00（+79.0%，≈1.79×FedAvg）"
    ]
}

# 生成 DataFrame
df = pd.DataFrame(data)

# 保存为 Excel 文件
df.to_excel("federated_learning_results.xlsx", index=False)
print("Excel 文件已生成：federated_learning_results.xlsx")