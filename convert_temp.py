try:
    with open("app/app_rohin_temp.py", "r", encoding="utf-16") as f:
        content = f.read()
except UnicodeError:
    with open("app/app_rohin_temp.py", "r", encoding="utf-8") as f:
        content = f.read()

with open("app/app_rohin_utf8.py", "w", encoding="utf-8") as f:
    f.write(content)
