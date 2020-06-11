from collections import Iterable

class Classmate(object):
    def __init__(self):
        self.names = list()

    def add(self, name):
        self.names.append(name)

    def __iter__(self):
        # 返回迭代器对象
        return ClassIterator(self)

class ClassIterator(object):
    def __init__(self, obj):
        self.obj = obj
        self.cur = 0

    # 包含__iter__方法的对象成可迭代iterable
    def __iter__(self):
        pass

    # 包含__next__的方法对象成迭代器iterator
    def __next__(self):
        if self.cur < len(self.obj.names):
            res = self.obj.names[self.cur]
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
