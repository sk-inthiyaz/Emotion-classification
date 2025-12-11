try:
    with open("app/app_rohin_temp.py", "r", encoding="utf-16") as f:
        print(f.read())
except UnicodeError:
    with open("app/app_rohin_temp.py", "r", encoding="utf-8") as f:
        print(f.read())
