with open("example.txt", "r") as f:
        print(f.read())

with open("example.txt", "r") as f:
        f.linereads = f.readlines()
        print(f.linereads)