# import numpy as np
#
#
# def generalized_fizzbuzz(start, end, divisor1, divisor2, word1, word2):
#     nums = np.array(range(start, end + 1))
#     for i in nums:
#         msg = ""
#         if i % divisor1 == 0:
#             msg = msg + word1
#         if i % divisor2 == 0:
#             if msg == word1:
#                 msg = msg + " " + word2  # remove the " " for "word1word2" instead of "word1 word2" (with the space)
#             else:
#                 msg = msg + word2
#         if msg == "":
#             msg = i
#         print(msg)
#
# #test case
# generalized_fizzbuzz(3, 16, 3, 5, "Fizz", "Buzz")


nums1 = [2, 3, 4, 5, 6, 10, 14, 15, 16, 17, 18, 24, 28, 30, 36]
divisors1 = [2, 3, 4, 6]
words1 = ["Alpha", "Beta", "Gamma", "Delta"]
word_priority1 = [3, 4, 1, 2]
new_divisors1 = []
new_words1 = []

#try a dict?

for i in word_priority1:
    new_divisors1.append(divisors1[i - 1])

print(new_divisors1)

for i in word_priority1:
    new_words1.append(words1[i - 1])

print(new_words1)



msg = ""
for i in nums1:
    for j in new_divisors1:
        if i % j == 0:
            msg = msg + new_words1[j]


# def hypergeneralized_fizzbuzz(nums, divisors, words, word_priority):
#     new_divisors = []
#     for i in word_priority:
#         new_divisors.append(divisors[i - 1])
#     for i in nums:
#         msg = ""
#         for j in new_divisors:
#             if i % j == 0:
#                 msg = msg + words[j]
#     print(msg)
#
# hypergeneralized_fizzbuzz(nums1 , divisors1, words1, word_priority1)