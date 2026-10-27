# 文本向量 Embedding

主题：Agent 开发学习资料

Embedding 会把文本转换为一串浮点数向量。语义相近的文本在向量空间中通常距离更近。RAG 将文档 chunk 和用户问题使用同一 embedding 模型向量化，再按相似度找出相关 chunk。文档和查询必须使用相同模型与维度。
