# 增量摘要

主题：Agent 开发学习资料

摘要表可记录 summary 和 covered_until_id。covered_until_id 表示摘要已经覆盖到哪条历史消息。new_need_summarize 是旧消息中 id 大于 covered_until_id 的部分，因此每轮只摘要新增进入旧消息区的内容，而不必重复摘要所有历史。
