import sys

data = sys.stdin.read()

for l in data.split('\n'):
    cur_line = l.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')
    if len(cur_line) < 4:
        continue
    layers = cur_line[3:]
    for i, cur_layer in enumerate(layers):
        if len(cur_layer.replace(' ', '')) == 0:
            continue
        last = int(int(cur_line[1]) == i + 1)
        first = int(0 == i)
        mid = i * (last != 1) * (first != 1)
        inp = cur_line[0]
        #print(1/float(inp)**4, *cur_line[1:3], cur_layer, first, last, mid)
        print(float(inp), *cur_line[1:3], cur_layer, first, last, mid)
