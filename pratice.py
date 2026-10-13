import json

def greet(name: str) -> str:
    return f"Hello, {name}!"
print(greet("上官亮"))

try:
    num=int("abc")
except ValueError as e:
    print(f"Error: {e}")

d={"name": "上官亮", "age": 30, "city": "New York"}
s=json.dumps(d,ensure_ascii=False)
print(s,type(s))
d2=json.loads(s)
print(d2,type(d2))