python测试框架nose研究

赘肉侠客 2015-07-20 22:50:37   7858   收藏
分类专栏： tempest 文章标签： 自动化测试 nose tempest
版权
最近在使用openstack自动化测试工具tempest，tempest中用到了nose，这里先了解一下nose的用法吧。

关于nose的安装什么的就不介绍了，这个可以在网上搜一搜。

这里主要介绍nose框架和如何使用。好吧，也是烂大街的东西，这里全当总结汇总下吧。



第一部分：

这一部分应该写写nose的前世今生，祖宗十八代什么的，可惜找了下没看到比较详细的介绍，可能大家都觉着知道是个啥，怎么用就行了吧。好吧，me，too~

python的单元测试有很多，tempest里面就涉及到了nose，unittest，testtools。这三个是啥关系呢？好吧，就是基友关系。

unittest是python单元测试的老祖宗，里面提供了很多单元测试的概念。比如testcase，testsuite，testloader，texttestrunner。

1、testcase 也就是测试用例，一般一个unittest的实例就是一个testcase，可以理解为测试的最小单元。

2、testsuite 测试套，就是多个测试case的集合。多个testcase加载后，然后testsuite编排执行顺序啥的。

3、testloader 测试加载器。啥意思呢，你写的测试用例要首先实例化了以后才能用，但是怎么实例化，实例化哪些这个就是testloader干的活了。它通过各种匹配原则来进行加载，创建测试用例的实例，然后返回。

4、texttestrunner，这个是真正执行测试用例的，执行之前testloader加载的那些实例。执行测试用例实例的run方法（和多线程那个run一个意思），然后用例就咔咔执行了。



然后testtools看他不爽，给它增强了一下。看了tempest里面的那个test.py，它是继承自testtools的，至于它具体干了啥，没仔细研究过。不过用例执行的执行结构啥的还是一样一样的。

好了，可以介绍下nose了。其实吧，看了tempest以后，大部分的还是继承自testtools，感觉nose在里面的作用就是个testloader，加载testcase的。不过使用nose用来加载还是很强大的，可以运行一个类、函数、文件、包什么的。好像是有个匹配规则，还没有看过nose的代码，这里就不瞎哔哔了。



看个栗子：

unittest的灵感来自junit，好吧，我也没用过。但是很明显的一个痕迹就是搞个unittest测试用例，你怎么也得写个类，继承自unittest，然后再写方法。比如下面这种：

```python
import unittest
 
 
class A(object):
 
    def __init__(self):
        pass
 
    def add(self, a, b):
        return a - b
 
 
class SimpleTest(unittest.TestCase):
 
    def setUp(self):
        self.a = A()
 
    def test_hehe(self):
        assert self.a.add(10, 9) == 1
 
if __name__ == '__main__':
    unittest.main()
```

if __name__ == '__main__':
    unittest.main()
其实上面这么多，有用的部分就是 assert self.a.add(10, 9) == 1，其他的都是准备工作。
为啥要搞这么麻烦呢？如果我可以直接把多余的省了，直接使用那句话不就行了。好吧，nose满足你，如果使用nose写个用例，最多也就下面这样吧。

```python
def test_hehe():
    a = A()
    assert self.a.add(10, 9) == 1
```

好了，不必折腾整个类什么的了。当然这个地方就举例说明下，肯定把nose的优点给写出来了。其实nose是支持多种形式的，比如一个测试函数，一个测试类什么的都可以。



第二部分：

这部分主要介绍nose的使用。目前使用的也只是最简单的方法：nosetests xxx，其他的还没有深入研究。

其中xxx可以是目录、包、文件等，nose可以识别这些东东中的测试用例并执行。

如果想单独执行一个类中的测试方法，可以使用nosetests xxxx.py:Testclass.testmethod 其中xxxx.py是文件名，Testclass是类名，testmethod是要执行的方法。



第三部分：

这部分主要介绍tempest中的具体应用。

黑盒测试的主要工作就是模拟各种场景对待测系统进行测试（貌似有个名字叫数据驱动测试）。所以测试框架要做的也是围绕着构造数据来展开的。

一个完整的用例应该包括：前置条件，测试过程执行，后置条件。这些unittest都已经为你考虑过了，这个就是setUp和teardown方法的由来了。

但是看了tempest中的一个test.py（算是所有测试的基类吧）后，对一个类中的setUp和setUpClass的执行不是很清楚。不说了，直接上简化后的代码搞一把试试。

```python
# encoding:utf-8
import testtools
 
 
class TestSetUp(testtools.TestCase):
 
    @classmethod
    def setUp(cls):
        print "setUp running"
 
    @classmethod
    def setUpClass(cls):
        print "setUpClass running"
 
    @classmethod
    def tearDown(cls):
        print "teardown running"
 
    @classmethod
    def tearDownClass(cls):
        print "teardownclass running"
 
    def test_func1(self):
        print "func1 running"
 
    def test_func2(self):
        print "func2 running"
 
    def test_func3(self):
        print "func3 running"


```

使用nosetests 执行，结果如下，其中-s是会打印具体输出，不加的话就看不到输出了：



从上面的输出可以看出执行步骤：setupClass --> setup -->testfunc-->teardown -->setup -->testfunc-->teardown ...

setUpclass相当于全局的前置条件，teardownclass相当于全局的后置条件，setUp相当于每个用例的前置条件，teardown相当于每个用例的后置条件。

所以tempest中初始化那些请求的client都是在setupclass中的resource_setup中进行的，然后在用例的整个运行周期中都有效。
