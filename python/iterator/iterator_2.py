from collections import Iterable
import time

class Classmate(object):
    def __init__(self):
        self.names = list()
        self.cur = 0

    def add(self, name):
        self.names.append(name)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur < len(self.names):
            res = self.names[self.cur]
            self.cur += 1
            return res
        else:
            raise StopIteration


if __name__ == '__main__':
    classmate = Classmate()
    classmate.add('Jerry')
    classmate.add('Annie')
    classmate.add('Sophie')
    print('Iterable: {}'.format(isinstance(classmate, Iterable)))

    for temp in classmate:
        print(temp)
        time.sleep(1)
