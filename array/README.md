# Python中的数组

Python中并没有严格意义上的“数组”类型，但是有几种数据结构可以用来实现类似数组的功能。  
最常用的是列表（list），它是一个动态数组，能够存储不同类型的对象，并且大小是可变的。  
此外，还有其他更专业的库和类型，例如NumPy库中的ndarray，它提供了类似于 C/C++ 数组的  
固定大小、同质（所有元素类型相同）的一维或多维数组。  
  
1、列表(list)  
特点：Python 的列表是动态的，可以在运行时改变大小，支持不同的数据类型。  
使用场景：当你需要一个灵活、多功能的容器来存储一组项目时，列表是非常好的选择。  
```python
example_list = [1, 2, 3, 'four', 5.0]  # 可以包含不同类型的数据
```
  
2、元组(tuple)  
元组的内容是不可变的，即一旦创建就不能修改其内容。因此，它们不完全等同于数组，  
但在某些情况下可以作为静态数组的替代品。  
```python
example_tuple = (1, 2, 3)  # 创建后不能修改
```
  
3、NumPy的ndarray  
如果你需要进行大量的数值计算或者处理大型多维数组，NumPy提供了高效的ndarray类型，这是Python标准库中没有的。  
特点：固定大小、同质（所有元素必须是相同的类型）、高效的数学运算。  
使用场景：科学计算、数据分析等领域。  
```python
import numpy as np
my_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)  # 创建一个一维数组，指定元素类型为32位整数
```
  
4、array 模块  
Python 还有一个内置的array模块，它提供了一个类似于列表的对象，但是只能存储固定类型的数值数据，如整数或浮点数。  
特点：节省空间，只存储单一类型的数值数据。  
使用场景：当你需要一个简单的数值数组并且想要比列表更高效地存储时。  
```python
import array
my_array = array.array('i', [1, 2, 3, 4, 5])  # 创建一个整数数组
```
  
代码分析：  
  
array_simulate_by_list.py文件使用list模拟数组。  
```python
my_list = [1, 2, 3, 4, 5]
```
初始化数组my_list，包括5个元素。  
```python
element = my_list[2]
```
访问数组index为2的元素。  
```python
my_list.insert(2, 'inserted')
```
在数组index为2的位置插入一个元素，它是其内容为'inserted'的字符串。原来的index 2及以后的元素依次后移一位。  
这一行执行后，my_list数组的内容变成[1, 2, 'inserted', 3, 4, 5]，共6个元素，index 2是字符串类型，其他为整型。  
可以看出，用list实现的数组是动态数组，其大小是可变的，并且其成员可以是不同类型。  
```python
my_list.remove('inserted')
```
按value删除数组元素，这里指定删除value为'inserted'的数组元素。执行完这一句后，数组的内容变成[1, 2, 3, 4, 5]。  
```python
del my_list[0]
```
按index删除数组元素，这里指定删除index为0的数组元素。执行完这一句后，数组的内容变成[2, 3, 4, 5]。  
```python
for item in my_list:
    print(item)
```
遍历数组。  
```python
if 3 in my_list:
    print("Found 3 at index:", my_list.index(3))
```
查看数组中是否有value为3的元素，如果存在，用my_list.index(3)取得value 3对应的索引。  
```python
my_list.append(6)
```
在数组末尾追加一个元素6。这一句执行完，数组内容变成[2, 3, 4, 5, 6]。  
```python
my_list.extend([7, 8, 9])
```
在数组末尾追加3个元素7, 8, 9。这一句执行完，数组内容变成[2, 3, 4, 5, 6, 7, 8, 9]。  
  
array_by_array_module.py文件利用array模块实现数组，与list类似，它也是动态数组，可以插入，删除，追加元素，但是它只能保存相同类型的数据。  
  
array_by_numpy_ndarray.py文件利用numpy.ndarray实现数组，与list类似，它也是动态数组，可以插入，删除，追加元素，但是它只能保存相同类型的数据。  
  
### License  
  
MIT
