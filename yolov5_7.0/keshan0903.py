import os

a = False
b = False
a |= b
print(a)

a = False
b = True
a |= b
print(a)


a = True
b = False
a |= b
print(a)

a = True
b = True
a |= b
print(a)

print("=====================")

a = False
b = False
a = a | b
print(a)

a = False
b = True
a = a | b
print(a)


a = True
b = False
a = a | b
print(a)

a = True
b = True
a = a | b
print(a)
