# Python 异步调用

主题：Agent 开发学习资料

async def 定义协程函数。调用协程函数会得到协程对象，只有 await 或 asyncio.run 才会真正执行。asyncio.create_task 可以并发创建任务，asyncio.gather 等待多个任务完成，asyncio.wait_for 为单个任务设置超时。
