import numpy as np


def kwadraty(input_list):
    output_list = []
    for i in input_list:
        if i >= 0:
            output_list.append(i * i)
    return output_list


input = [2, -3, 4, 9, -10.5, 0.04]

out = kwadraty(input)

print(out)
