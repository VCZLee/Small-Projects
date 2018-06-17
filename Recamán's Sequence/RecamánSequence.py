import matplotlib.pyplot as plt
import turtle


starting_point = 0
starting_addend = 1
num_iterations = 1000
recaman_set = set()
recaman_list = []
recaman_set.add(starting_point)
#element in set lookup is O(1) compared to element in list lookup which is O(n)
recaman_list.append(starting_point)
for i in range(1, num_iterations):
    if starting_point - starting_addend < 0 or starting_point - starting_addend in recaman_set:
        starting_point = starting_point + starting_addend
    else:
        starting_point = starting_point - starting_addend
    starting_addend = starting_addend + 1
    recaman_set.add(starting_point)
    recaman_list.append(starting_point)

# print(recaman_list)
#
# plt.scatter(range(1,num_iterations + 1), recaman_list)
# plt.show()



integer_scale = 2
sizing = max(recaman_list) * integer_scale

screen = turtle.Screen()
screen.screensize(10000, 10000)

t = turtle.Pen()
t.speed(0)
t.hideturtle()

t.right(90)
for i in range(0, num_iterations - 1):
    factor = integer_scale * (i + 1)
    if i % 2 == 1:
        factor = factor * -1
    if recaman_list[i+1] - recaman_list[i] < 0:
        factor = factor * -1
    t.circle(factor, 180)
turtle.done()