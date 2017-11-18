import numpy as np


def generalized_fizzbuzz(start, end, divisor1, divisor2, word1, word2):
    nums = np.array(range(start, end + 1))
    for i in nums:
        msg = ""
        if i % divisor1 == 0:
            msg = msg + word1
        if i % divisor2 == 0:
            if msg == word1:
                msg = msg + " " + word2  # remove the " " for "word1word2" instead of "word1 word2" (with the space)
            else:
                msg = msg + word2
        if msg == "":
            msg = i
        print(msg)


# test case
generalized_fizzbuzz(3, 16, 3, 5, "Fizz", "Buzz")


def hypergeneralized_fizzbuzz(nums, divisors, words, word_priority):
    new_divisors = []
    divisors_dict = {}
    for counter, element in enumerate(divisors):
        divisors_dict.update({element: words[counter]})
    for i in word_priority:
        new_divisors.append(divisors[i - 1])
    for i in nums:
        msg = ""
        for j in new_divisors:
            if i % j == 0:
                msg = msg + divisors_dict[j]
        if msg == "":
            msg = i
        print(msg)


#test case
nums1 = [1, 2, 3, 4, 5, 6, 10, 12, 14, 15, 16, 20, 24, 30, 35, 36]
divisors1 = [2, 3, 4, 6]
words1 = ["Alpha", "Beta", "Gamma", "Delta"]
word_priority1 = [3, 4, 1, 2]
hypergeneralized_fizzbuzz(nums1, divisors1, words1, word_priority1)
