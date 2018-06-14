import matplotlib.pyplot as plt

starting_point = 0
starting_addend = 1
num_iterations = 500000
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
plt.scatter(range(1,num_iterations + 1), recaman_list)
plt.show()