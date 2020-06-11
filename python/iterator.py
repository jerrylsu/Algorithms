from collections import Iterable

class Classmates(object):
    def __init__(self):
        self.names = list()

    def add(self, name):
        self.names.append(name)

    def __iter__(self):
        return ClassmatesIterator(self)

class ClassmatesIterator(object):
    def __init__(self, obj):
        self.obj = obj
        self.cur = 0

    def __iter__(self):
        pass

    def __next__(self):
        if self.cur < len(self.obj.names):
            res = self.obj.names[self.cur]
            self.cur += 1
            return res


if __name__ == '__main__':
    classmate = Classmates()
    classmate.add('Jerry')
    classmate.add('Annie')
    classmate.add('Sophie')
    print('Iterable: {}'.format(isinstance(classmate, Iterable)))

    for temp in classmate:
        print(temp)
