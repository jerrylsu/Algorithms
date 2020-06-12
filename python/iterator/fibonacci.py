class Fibonacci(object):
    def __init__(self, num):
        self.num = num
        self.cur = 0
        self.a = 0
        self.b = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur < self.num:
            res = self.a
            self.a, self.b = self.b, self.a + self.b
            self.cur += 1
            return res
        else:
            raise StopIteration


if __name__ == '__main__':
    fibo = Fibonacci(10)
    for num in fibo:
        print(num)
