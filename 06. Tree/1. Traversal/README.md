## 如何遍历树
有两种策略遍历树：

广度优先搜索
Breadth First Search (BFS)

我们按照高度顺序从上到下逐层扫描树。更高level的节点将在具有较低level的节点之前被访问。

深度优先搜索
Depth First Search (DFS)
在这个策略中，我们采用深度作为优先级，以便从一个根开始，一直到达某个叶子，然后回到root到达另一个分支。
根据根节点，左节点和右节点之间的相对顺序，DFS策略可以进一步区分为先序，中序和后序。

On the following figure the nodes are numerated in the order you visit them, please follow 1-2-3-4-5 to compare different strategies.
