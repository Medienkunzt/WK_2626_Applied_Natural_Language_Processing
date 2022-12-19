import re

m = re.search(r'(?<=abc)def', 'abcdef def')
print(m)
print(m.group(0))

print(re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest'))

x = re.findall(r'(\w+)=(\d+)', 'set width=20 and height=10')
for i in x:
    print(i)

for i in re.finditer(r'(\w+)=(\d+)', 'set width=20 and height=10', flags=0):
    print(i.group(0))

print('abcd \n 2134')
print(r'abcd \n 2134')

x = re.findall(r'(fall)(s|acy)?', 'fallacy fall falls fell fee')
print(x)
for i in x:
    print(i)

m = re.search(r'fall(s|acy)?', 'fallacy fall falls fell fee')
print(m)
