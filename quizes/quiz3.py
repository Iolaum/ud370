print("Doing it their way")

# b = 1000000000
b = 1
a = b
for i in range(1000000):
    b = b + 0.000001

print(b - a)
