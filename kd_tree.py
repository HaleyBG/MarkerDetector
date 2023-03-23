# 2022/3/31
# 16:11
import pickle
# kd树结点
class Node:
    def __init__(self):
        # 左孩子
        self.left = None
        # 右孩子
        self.right = None
        # 父节点
        self.parent = None
        # 特征坐标，切分的坐标点（坐标集合中中间那个坐标）
        self.x = None
        # 切分轴
        self.dimension = None
        # 是否被访问过，标记
        self.flag = False



def clear_flag(node):
    """
    将标记清零
    :param node: Node
    :return:
    """
    node.flag=False
    if node.left:
        clear_flag(node.left)
    if node.right:
        clear_flag(node.right)
# 构建kd树
def construct(d, data, node, layer):
    """
    :type d: int
    d是向量的维数
    :type data: list
    data是所有向量构成的列表
    :type node: Node
    node是当前进行运算的结点
    :type layer: int
    layer是当前kd树所在层数
    """
    node.dimension = layer % d  # 防止维数越界
    # 如果只有一个元素，说明到了叶子结点，该分支结束
    if len(data) == 1:
        node.x = data[0]  # 该data中间那个就是唯一的一个坐标
        return
    if len(data) == 0:  # 没有代表的数据就作为一个空叶子结点
        return
    # 1,data中的数据按layer%N维进行排序
    data.sort(key=lambda x: x[layer % d])  # 表示按list中元素的第layer%d维进行排列，此处使用了匿名函数，key是用来比较的元素
    # 2,计算中间点的索引，偶数则取中间两位中较大的一位,记为该结点的特征坐标
    middle = len(data) // 2  # 除法取整
    node.x = data[middle]
    # 3，划分data
    dataleft = data[:middle]
    dataright = data[middle + 1:]
    # 4,左孩子结点

    left_node = Node()
    node.left = left_node
    left_node.parent = node
    construct(d, dataleft, left_node, layer + 1)
    # 5，右孩子结点

    right_node = Node()
    node.right = right_node
    right_node.parent = node
    construct(d, dataright, right_node, layer + 1)


def distance(a, b):  # 计算欧式距离
    """
    :type a: list
    :type b: list
    """
    dis = 0
    for i in range(0, len(a)):
        dis += (a[i] - b[i]) ** 2
    return dis ** 0.5


def change_L(L, x, p, K):  # 判断并进行是否将该点加入近邻点列表
    """
    :type L: list
    L是近邻点列表
    :type x: list
    x是判断是否要加入近邻列表的向量
    :type p: list
    p是目标向量
    :type K:int
    K是近邻列表的最大元素个数
    """
    if len(L) < K:
        L.append(x)
        return
    dislist = []
    for i in range(0, K):
        dislist.append(distance(p, L[i]))
    index = dislist.index(max(dislist))
    if distance(p, x) < dislist[index]:  # 若x和p之间的距离小于L到p中最远的点，就用x替换此最远点
        L[index] = x
    return max(dislist)


# 搜索kd树
def search(node, p, L, K):
    """
    :type List: list
    :type node: Node
    类Node，整个树的框架，里面包含父子结点信息，以及每个父子结点含有的坐标点
    :type p: list
    目标坐标
    :type L: list
    L为有k个座位的列表，用于保存已搜寻到的最近点
    :type K: int
    K为近邻个数
    :type L0: list
    :type f: bool
    """

    # 1，根据p的坐标值和每个点的切分轴向下搜索,先到达底部结点
    n = node  # 用n来记录结点的位置，先从顶部开始,直到叶子结点，循环完的n为叶子节点
    while True:
        # 若到达了叶子结点则退出循环
        if (n.left == None) & (n.right == None):
            break
        if n.x[n.dimension] > p[n.dimension]:
            n = n.left
        else:
            n = n.right

    n.flag = True  # 标记为已访问过
    if n.x is None:  # 若为空叶子结点，则不必记录数值
        pass
    else:
        change_L(L, n.x, p, K)  # 若符合插入条件，就插入，不符合就不插入

    # 从叶子节点往上找到了根节点
    while True:
        # 若当前结点是根结点则输出L算法完成
        if n.parent is None:
            if len(L) < K:
                print('K值超过数据总量')
            return L
        # 当前结点不是根结点，向上爬一格
        else:
            n = n.parent
            while n.flag == True:
                # 若当前结点被访问过，就一直向上爬，到没被访问过的结点为止
                # 若向上爬时遇到了已经被访问过的根结点，说明另一边已经搜索过了搜索结束
                if (n.parent is None) & (n.flag):
                    if len(L) < K:
                        print('K值超过数据总量')
                    return L
                n = n.parent
            # 此时n未被访问过,将其标记为访问过
            n.flag = True

            # 如果此时 L 里不足 k 个点，则将节点特征加入 L；
            # 如果 L 中已满 k 个点，且当前结点与 p 的距离小于与L的最大距离，
            # 则用节点特征替换掉 LL 中离最远的点。
            change_L(L, n.x, p, K)
            ''' 计算p和当前节点切分线的距离。如果该距离小等于于 LL 中最远的距离或者 LL 中不足 kk 个点，
                        则切分线另一边或者 切分线上可能有更近的点，
                        因此在当前节点的另一个枝从 (一) 开始执行。'''
            dislist = []
            for i in range(0, len(L)):
                dislist.append(distance(p, L[i]))
            if (abs(p[n.dimension] - n.x[n.dimension]) < max(dislist)) | (len(L) < K):
                if n.left.flag == False:
                    return search(n.left, p, L, K)
                else:
                    return search(n.right, p, L, K)
            # 如果该距离大于等于 L 中距离 p 最远的距离并且 L 中已有 k 个点，则在切分线另一边不会有更近的点，重新执行(三)


# 使用说明
# data表示数据集，这里是list类型，元素表示数据点，是d维向量，d表示data中数据点的维度，p为要寻找k近邻的点，K为近邻个数，其他均为默认值
if __name__=='__main__':
    with open('candidate.pkl','rb') as f:
        data = pickle.load(f)
    # data = [[5, 4], [7, 2], [2, 3], [4, 7], [8, 1], [9, 6]]
    node = Node()
    construct(d=2, data=data, node=node, layer=0)
    print(search(node=node, p=[44,6], L=[], K=5))


