import struct

def open_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    print(data)

def run():
    name = input("filename: ")
    open_file(name)

run()