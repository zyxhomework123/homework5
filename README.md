# homework5
# 代码结果
# 01 数据加载与预处理
<img src="![66b20fc0cf68d0f28fa0dc2d23cb01f](https://github.com/user-attachments/assets/51618749-d127-4075-94bc-78c66de8ca72)
" />

# 模型初始化
<img src="![52c1ab7edee20d130d4330c44af4e0b](https://github.com/user-attachments/assets/c898cca9-7a89-4a12-b178-ec0b985746d2)
" />

# RNN序列处理验证
<img src="![028b3fae83afac1772b61227e135492](https://github.com/user-attachments/assets/31c23944-f33f-445f-842c-8267800b3cf6)
" />

# 模型性能评估
<img src="![30b21716cf8f3081fcbc44c49f76921](https://github.com/user-attachments/assets/f88540dc-bc9c-4e0b-988c-6b87bd9eae16)
" />

# 推理能力验证
<img src="![bae293907e615def0ca637673b4c6a4](https://github.com/user-attachments/assets/69e32c4e-a73b-42db-b08b-b9e992bc959a)
" />
<img src="![579c0d80dcbc82e982125f7bc3a9d73](https://github.com/user-attachments/assets/557d5bf3-82ce-4307-bc26-ddda9eaa39da)
" />

# 无条件姓氏生成
<img src="![23bd66b173d30a0af366082c89bb89c](https://github.com/user-attachments/assets/37c3ca82-ba7e-4cf9-9db5-b673dd238a9f)
" />

# 有条件姓氏生成
<img src="![4c59181b2d845c2fa6f66aaa122fc88](https://github.com/user-attachments/assets/8c1ce412-ae3f-42cc-9bf1-3ba7080d4bdc)
" />


# 问题与答案

## ① 两个模型的核心差异体现在什么机制上？
**答案：B. 是否考虑国家信息作为生成条件**

---

## ② 在条件生成模型中，国家信息通过什么方式影响生成过程？
**答案：B. 作为GRU的初始隐藏状态**

---

## ③ 文件2中新增的 `nation_emb` 层的主要作用是？
**答案：B. 将国家标签转换为隐藏状态初始化向量**


---

## ④ 对比两个文件的 `sample_from_model` 函数，文件2新增了哪个关键参数？
**答案：B. nationalities**
