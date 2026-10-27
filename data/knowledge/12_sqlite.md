# SQLite 基础

主题：Agent 开发学习资料

SQLite 是嵌入式数据库，数据保存在本地单个文件中。表由列和约束组成。联合主键 PRIMARY KEY(user_id, preference_key) 表示这两个字段的组合必须唯一。INSERT ON CONFLICT 可以在主键重复时更新已有记录。
