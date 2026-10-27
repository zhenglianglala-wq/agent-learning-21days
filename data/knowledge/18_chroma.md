# Chroma 向量数据库

主题：Agent 开发学习资料

Chroma 可以保存文本、向量、唯一 id 和 metadata。metadata 可保存 filename、chunk_id 和主题。查询时提供问题向量和 n_results，Chroma 返回最相近的片段及其 metadata。持久化目录可让索引在程序重启后继续存在。
