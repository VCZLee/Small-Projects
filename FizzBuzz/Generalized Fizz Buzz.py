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

#test case
generalized_fizzbuzz(3, 16, 3, 5, "Fizz", "Buzz")

