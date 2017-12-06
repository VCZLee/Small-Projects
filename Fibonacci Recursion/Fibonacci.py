def fibo(n):
    if n < 1 or isinstance(n, int) == False:
        raise Exception("Input value must be an integer greater than or equal to 1")
    if n == 1:
        return 0
    if n == 2:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)

fibodict = {}
def fibom(n):
    if n < 1 or isinstance(n, int) == False:
        raise Exception("Input value must be an integer greater than or equal to 1")
    if n == 1:
        return 0
    if n == 2:
        return 1
    if n in fibodict:
        return fibodict[n]
    else:
        fibodict[n] = fibom(n - 1) + fibom(n - 2)
        return fibodict[n]

class Fibonacci:
    def __init__(self,num1,num2):
        self.num1 = num1
        self.num2 = num2 # num1 and num2 set the starting points for the fibonacci sequence
        self.fibonacci_dict = {}

    def sequence_number(self,n):
        if n < 1 or isinstance(n, int) == False:
            raise Exception("Input value must be an integer greater than or equal to 1")
        if n == 1:
            return self.num1
        if n == 2:
            return self.num2
        if n in self.fibonacci_dict:
            return self.fibonacci_dict[n]
        else:
            self.fibonacci_dict[n] = self.sequence_number(n - 1) + self.sequence_number(n - 2)
            return self.fibonacci_dict[n]


print(Fibonacci(0,1).sequence_number(70))
print(Fibonacci(-4,0.964).sequence_number(70))