#### 56. Merge Intervals

```python3
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals: return []
        intervals = sorted(intervals, key=lambda x: x[0])
        res = []
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            cur = intervals[i]
            last = res[-1]
            if cur[0] <= last[1]:
                last[1] = max(cur[1], last[1])
            else:
                res.append(cur)
        return res
```

#### 54. Spiral Matrix

```python3
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        top, left = 0, 0
        bottom, right = len(matrix) - 1, len(matrix[0]) - 1
        while top <= bottom and left <= right:
            if top == bottom:
                for i in range(left, right + 1):
                    res.append(matrix[top][i])
            elif left == right:
                for i in range(top, bottom + 1):
                    res.append(matrix[i][left])
            else:
                for i in range(left, right):
                    res.append(matrix[top][i])
                for i in range(top, bottom):
                    res.append(matrix[i][right])
                for i in range(right, left, -1):
                    res.append(matrix[bottom][i])
                for i in range(bottom, top, -1):
                    res.append(matrix[i][left])
            top += 1
            left += 1
            bottom -= 1
            right -= 1
        return res
```

#### 88. Merge Sorted Array

```python3
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        l1, l2, end = m - 1, n - 1, m + n -1
        while l1 >= 0 and l2 >= 0:
            if nums2[l2] >= nums1[l1]:
                nums1[end] = nums2[l2]
                l2 -= 1
            else:
                nums1[end] = nums1[l1]
                l1 -= 1
            end -= 1
        if l2 >= 0:
            nums1[:end+1] = nums2[:l2+1] 
```

#### 7. Reverse Integer

```python3
class Solution:
    def reverse(self, x: int) -> int:
        retval = int(str(abs(x))[::-1])
        
        if(retval.bit_length()>31):
            return 0
    
        if x<0:
            return -1*retval
        else:
            return retval
```

#### 8. String to Integer (atoi)

```python3
class Solution:
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.strip()
        if str == '': return 0
        ls = list(str)
        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['-', '+']: del ls[0]
        result = 0
        for i, char in enumerate(ls):
            if not ls[i].isdigit(): break
            result = result * 10 + (ord(ls[i]) - ord('0'))
        return min(max(result * sign, -2**31), 2**31 - 1)
        
    from functools import reduce
		def str2float(s):
    		def fn(x,y):
            return x*10+y
    		n=s.index('.')
    		s1=list(map(int,[x for x in s[:n]]))
    		s2=list(map(int,[x for x in s[n+1:]]))
    		return reduce(fn,s1)+reduce(fn,s2)/10**len(s2)
```

#### 20. Valid Parentheses

```python3
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        mapping = {'(': ')', '{': '}', '[': ']'}
        for c in s:
            if c in mapping:
                stack.append(c)
            elif not stack or mapping[stack.pop()] != c:
                return False
        return not stack
```

### strStr

```python3
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        haystack_len = len(haystack)
        needle_len = len(needle)
        if needle == '':
            return 0
        for i in range(haystack_len - needle_len + 1):
            if haystack[i:i+len(needle)] == needle:
                return i       
        return -1
```

## Sort

#### Top Sort

```python3
# Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

class Solution:
    """
    @param graph: A list of Directed graph node
    @return: A list of integer
    """
    def topSort(self, graph):
        # 1. 统计结点入度
        indegree = self.get_indegree(graph)
        # 2. BFS
        order = []
        start_nodes = [n for n in graph if indegree[n] == 0] # 入度为0的所有结点
        queue = collections.deque(start_nodes)       # 队列中存储的是入度为0的点
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in node.neighbors:          # 遍历该节点的所有邻居节点，第一层遍历。
                indegree[neighbor] -= 1      # 将队列中输出的点的所以临界点入度减1
                if indegree[neighbor] == 0:  # 将入度为0的结点放入队列
                    queue.append(neighbor)         
        return order

    def get_indegree(self, graph):
        """ 计算每一个结点的入度数
        """
        indegree = {x: 0 for x in graph}    # 初始化每一个结点的入度数为0
        for node in graph:
            for neighbor in node.neighbors:
                indegree[neighbor] += 1
        return indegree
```

## Two Points

####  633. Sum of Square Numbers

```python3
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        nums = [i**2 for i in range(int(c**0.5)+1)]
        low, hight = 0, len(nums) - 1
        while low <= hight:
            target = nums[low] + nums[hight] 
            if target < c:
                low += 1
            elif target > c:
                hight -= 1
            else:
                return True
        return False
```

#### 15. 3Sum

```python3
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:  # skip duplicate for i
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                s = nums[l] + nums[r] + nums[i]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                elif s == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]: l += 1 # skip
                    while l < r and nums[r] == nums[r-1]: r -= 1 # skip
                    l += 1
                    r -= 1
        return res
```

#### 18. 4Sum

```python3
    class Solution:
        def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
            n = len(nums)
            if n < 4: return []
            nums.sort()
            res = []
            for i in range(n-3):
                if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target: break     # 加速 (包括当前数在内和最小)
                if nums[i] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target: continue  # 加速（包括当前数在内和最大）
                if i > 0 and nums[i] == nums[i - 1]: continue
                for j in range(i + 1, n - 2):
                    if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target: break     # 加速
                    if nums[i] + nums[j] + nums[n - 2] + nums[n - 1] < target: continue  # 加速
                    if j > i + 1 and nums[j] == nums[j - 1]: continue
                    l, r = j + 1, n - 1
                    while l < r:
                        temp = nums[i] + nums[j] + nums[l] + nums[r]
                        if temp == target:
                            res.append([nums[i], nums[j], nums[l], nums[r]])
                            l += 1
                            r -= 1
                            while l < r and nums[l] == nums[l - 1]: l += 1
                            while l < r and nums[r] == nums[r + 1]: r -= 1
                        elif temp > target:
                            r -= 1
                        else:
                            l += 1
            return res
```

#### 704. Binary Search

```python3
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = left + ((right - left) >> 1)
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid
            elif nums[mid] > target:
                right = mid
        if nums[left] == target:
            return left
        if nums[right] == target:
            return right
        return -1
```

#### 33. Search in Rotated Sorted Array

```python3
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = left + ((right - left) >> 1)
            if nums[mid] == target:
                return mid
            if nums[mid] > nums[left]:
                if nums[left] <= target < nums[mid]:
                    right = mid
                else:
                    left = mid
            elif nums[mid] < nums[left]:
                if nums[mid] < target <= nums[right]:
                    left = mid
                else:
                    right = mid
        if nums[left] == target:
            return left
        if nums[right] == target:
            return right
        return -1
```

##  Linked List

#### 160. Intersection of Two Linked Lists

```python3
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if not headA or not headB: return None
        lenA, lenB = 0, 0
        curA, curB = headA, headB
        while curA:
            lenA += 1
            curA = curA.next
        while curB:
            lenB += 1
            curB = cuwwwrB.next
        curA, curB = headA, headB
        for _ in range(abs(lenA - lenB)):
            if lenA > lenB:
                curA = curA.next
            else:
                curB = curB.next
        while curA is not curB:
            curA = curA.next
            curB = curB.next
        return curA
```

#### 206. Reverse Linked List

```python3
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class Solution:
    def reverseLinkedList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        pre, cur, post = None, head, head.next
        while cur:
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post
        return pre
```

#### 21. Merge Two Sorted Lists

```python3

class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 and not l2:
            return None
        if not l1:
            return l2
        if not l2:
            return l1
        dummpy = cur = ListNode(-1)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummpy.next
```

#### 234. Palindrome Linked List

```python3
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        fast = head
        slow = self.reverseLinkedList(slow)
        while slow:
            if fast.val != slow.val:
                return False
            fast = fast.next
            slow = slow.next
        return True

    def reverseLinkedList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post
        return pre
```

#### 328. Odd Even Linked List

```python3
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        dummpy1 = odd = ListNode(-1)
        dummpy2 = even = ListNode(-1)
        i = 1
        while head:
            if i % 2:
                odd.next = head
                odd = odd.next
            else:
                even.next = head
                even = even.next
            head = head.next
            i += 1
        odd.next = dummpy2.next
        even.next = None
        return dummpy1.next
```

#### 19. Remove Nth Node From End of List

```python3
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummpy = slow = fast = ListNode(-1)
        fast.next = head
        for _ in range(n + 1):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummpy.next

```

#### 23. Merge k Sorted Lists

```python3
class Solution:
    def mergeKLists(self, lists):
        if not lists:
            return None
        
        if len(lists) == 1:
            return lists[0]
        
        mid = len(lists) // 2
        
        l = self.mergeKLists(lists[:mid])
        r = self.mergeKLists(lists[mid:])
        
        return self.merge2Lists(l, r)
    
    def merge2Lists(self, l, r):
        dummpy = cur = ListNode(-1)
        while l and r:
            if l.val < r.val:
                cur.next = l
                l = l.next
            else:
                cur.next = r
                r = r.next
            cur = cur.next
        cur.next = l if l else r
        return dummpy.next
```

## Tree

#### 104. Maximum Depth of Binary Tree

```python3
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # end condition
        if not root:
            return 0
        # divide & conquer
        leftDepth = self.maxDepth(root.left)
        rightDepth = self.maxDepth(root.right)
        res = max(leftDepth, rightDepth) + 1
        # return result
        return res
```

#### 110. Balanced Binary Tree

```python3
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        return self.helper(root) != -1
        
    def helper(self, root):
        if not root:
            return 0
        left = self.helper(root.left)
        if left == -1:
            return -1
        right = self.helper(root.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
```

#### 543. Diameter of Binary Tree

```python3
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.diameter = 0 
        self.helper(root)
        return self.diameter
        
    def helper(self, root):
        if not root:
            return 0
        left = self.helper(root.left)
        right = self.helper(root.right)
        self.diameter = max(self.diameter, left + right)
        return 1 + max(left, right)
```

#### 102. Binary Tree Level Order Traversal

```python3
from collections import deque

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        res = []
        queue = deque()
        queue.append(root)
        while queue:
            cur_level = []
            n = len(queue)
            for _ in range(n):
                node = queue.popleft()
                cur_level.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(cur_level)
        return res
```

#### 103. Binary Tree Zigzag Level Order Traversal

```python3
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        queue = collections.deque()
        queue.append(root)
        level = 0
        while queue:
            level_cur = []
            n = len(queue)
            for _ in range(n):
                node = queue.popleft()
                level_cur.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(level_cur if level % 2 == 0 else level_cur[::-1])
            level += 1
        return res
```

#### 105. Construct Binary Tree from Preorder and Inorder Traversal

```python3
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int], tree: TreeNode = None) -> TreeNode:      
        # Check if the root exist in inorder
        if len(preorder) and preorder[0] in inorder:
        
            # Find the root index
            root_index = inorder.index(preorder[0])
        
            # Make the root node
            tree = TreeNode(val=inorder[root_index])

            # Build left subtree
            tree.left = self.buildTree(preorder[1:root_index+1], inorder[:root_index])

            # Build right subtree
            tree.right = self.buildTree(preorder[root_index+1:], inorder[root_index+1:])
            
        return tree
```

## Search

### DFS

#### 695. Max Area of Island

```python3
class Solution:
    """O(M*N)：所有节点只遍历一次。
    M*N个节点；每个节点有4条边。
    """
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid: return 0
        row, col = len(grid), len(grid[0])
        max_area = 0
        for i in range(row):
            for j in range(col):
                max_area = max(max_area, self.dfs(grid, i, j))
        return max_area

    def dfs(self, grid, i, j):
        if i < 0 or i >= len(grid) \
        or j < 0 or j >= len(grid[0]) \
        or grid[i][j] == 0:
            return 0
        grid[i][j] = 0  # do choice, mark visited
        return 1 + self.dfs(grid, i - 1, j) \
                 + self.dfs(grid, i + 1, j) \
                 + self.dfs(grid, i, j - 1) \
                 + self.dfs(grid, i, j + 1)
```

#### 547. Number of Provinces

```python3
class Solution:
    """O(N*N)
    N个节点；每个节点至少1条边（只与自己相连），最多N条边（与所有节点相连）。
    """
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        node = len(isConnected)
        visited = [False] * node
        count = 0
        for i in range(node):
            if visited[i] == False:
                self.dfs(isConnected, i, visited)
                count += 1
        return count

    def dfs(self, isConnected, i, visited):
        visited[i] = True
        for j in range(len(isConnected)):
            if isConnected[i][j] == 1 and visited[j] == False:
                self.dfs(isConnected, j, visited)
```

#### 417. Pacific Atlantic Water Flow

```python3
class Solution:
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        row, col = len(heights), len(heights[0])
        pa_status = [[False for _ in range(col)] for _ in range(row)]
        al_status = [[False for _ in range(col)] for _ in range(row)]

        for r in range(row):
            self.dfs(heights, r, 0, pa_status)
            self.dfs(heights, r, col-1, al_status)

        for c in range(col):
            self.dfs(heights, 0, c, pa_status)
            self.dfs(heights, row-1, c, al_status)

        results = []
        for i in range(row):
            for j in range(col):
                if pa_status[i][j] == True and al_status[i][j] == True:
                    results.append([i, j])
        return results

    def dfs(self, heights, r, c, status):
        if status[r][c] == True:
            return
        status[r][c] =True
        for direction in self.directions:
            r_new, c_new = r + direction[0], c + direction[1]
            if r_new >= 0 and r_new < len(heights) \
            and c_new >= 0 and c_new < len(heights[0]) \
            and heights[r_new][c_new] >= heights[r][c]:
                self.dfs(heights, r_new, c_new, status)
```

### Backtracking

#### 46. Permutations

```python3
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        results, track = [], []
        self.backtrack(nums, track, results)
        return results

    def backtrack(self, nums, track, results):
        if len(track) == len(nums):
            results.append(track.copy())
            return
        for num in nums:
            if num in track:
                continue
            track.append(num)
            self.backtrack(nums, track, results)
            track.remove(num)
```

#### 79. Word Search

```python3
class Solution:
    """
    Time: O(M*N)
    Space: O(M*N)
    """
    def exist(self, board: List[List[int]], word: str) -> bool:
        row, col = len(board), len(board[0])
        visited = [[False for _ in range(col)] for _ in range(row)]
        for i in range(row):
            for j in range(col):
                if self.backtrack(board, i, j, word, 0, visited):
                    return True
        return False

    def backtrack(self, board, i, j, word, index, visited):
        if i < 0 or i >= len(board) \
        or j < 0 or j >= len(board[0]) \
        or visited[i][j] == True \
        or board[i][j] != word[index]:
            return False
        if index == len(word) - 1 and board[i][j] == word[index]:
            return True
        visited[i][j] = True    # avoid visit agian
        res = self.backtrack(board, i - 1, j, word, index + 1, visited) \
           or self.backtrack(board, i + 1, j, word, index + 1, visited) \
           or self.backtrack(board, i, j - 1, word, index + 1, visited) \
           or self.backtrack(board, i, j + 1, word, index + 1, visited)
        visited[i][j] = False
        return res


class Solution:
    """
    Time: O(M*N)
    Space: O(1)
    """
    def exist(self, board: List[List[int]], word: str) -> bool:
        row, col = len(board), len(board[0])
        for i in range(row):
            for j in range(col):
                if self.backtrack(board, i, j, word, 0):
                    return True
        return False

    def backtrack(self, board, i, j, word, index):
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return False
        if board[i][j] == "#" or board[i][j] != word[index]:
            return False
        if index == len(word) - 1 and board[i][j] == word[index]:
            return True
        temp = board[i][j]
        board[i][j] = "#"    # avoid visit agian
        res = self.backtrack(board, i - 1, j, word, index + 1) \
           or self.backtrack(board, i + 1, j, word, index + 1) \
           or self.backtrack(board, i, j - 1, word, index + 1) \
           or self.backtrack(board, i, j + 1, word, index + 1)
        board[i][j] = temp
        return res
```

#### 51. N-Queens

```python3
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def backtrack(row):
            if row == n:
                result.append(board.copy())
                return
            for i in range(n):
                if i in col or row + i in pie or row - i in na:
                    continue
                col.add(i)
                pie.add(row + i)
                na.add(row - i)
                board.append('.' * i + 'Q' + '.' * (n-i-1))
                backtrack(row + 1)
                col.remove(i)
                pie.remove(row + i)
                na.remove(row - i)
                board.pop()
        col, pie, na = set(), set(), set()
        result = []
        board = []
        backtrack(0)
        return result
```

#### 37. Sudoku Solver

```python3
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        
        def backtrack():
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if board[i][j] == '.':
                        for k in range(1, 10):
                            if isValid(i, j, str(k)):
                                board[i][j] = str(k)
                                if backtrack():
                                    return True
                                else:
                                    board[i][j] = '.'
                        return False
            return True
        
        def isValid(row, col, c):
            for i in range(9):
                if board[row][i] != '.' and board[row][i] == c:
                    return False
                if board[i][col] != '.' and board[i][col] == c:
                    return False
                if board[3 * (row // 3) + i // 3][ 3 * (col // 3) + i % 3] != '.' \
                and board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c:
                    return False
            return True
        
        backtrack()
```



## DP

#### 1029. 两地调度

### 1-Dim DP

#### 70. Climbing Stairs

```python3
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1 or n == 2:
            return n
        dp = [0 for _ in range(n)]
        dp[0], dp[1] = 1, 2
        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
```

#### 198. House Robber

```python3
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        dp = [0 for _ in range(len(nums))]
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        return dp[-1]
```

#### 213. House Robber II

```python3
class Solution:
    def rob(self, nums: List[int]) -> int:
        def helper(nums):
            if len(nums) == 1:
                return nums[0]
            if len(nums) == 2:
                return max(nums)
            dp = [0 for _ in range(len(nums))]
            dp[0], dp[1] = nums[0], max(nums[0], nums[1])
            for i in range(2, len(nums)):
                dp[i] = max(dp[i-1], dp[i-2] + nums[i])
            return dp[-1]
        if len(nums) == 1:
            return nums[0]
        if len(nums) <= 3:
            return max(nums)
        # helper(nums[1:-1]) 首尾都不抢的case省略，一定是最小的。
        return max(helper(nums[1:]), helper(nums[:-1]))
```



#### 413. Arithmetic Slices

```python3
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return 0
        dp = [0 for _ in range(len(nums))]
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = dp[i-1] + 1
        return sum(dp)
```

### 2-Dim DP

#### 64. Minimum Path Sum

```python3
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        dp = [[0 for _ in range(col)] for _ in range(row)]
        dp[0][0] = grid[0][0]
        for i in range(1, row):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for i in range(1, col):
            dp[0][i] = dp[0][i-1] + grid[0][i]
        for i in range(1, row):
            for j in range(1, col):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

#### 221. Maximal Square

```python3
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        row, col = len(matrix), len(matrix[0])
        dp = [[int(matrix[i][j]) for j in range(col)] for i in range(row)]
        max_side = max(max(row) for row in dp) 
        for i in range(1, row):
            for j in range(1, col):
                if matrix[i][j] == '1':
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]) + 1
                    max_side = max(dp[i][j], max_side)
        return max_side**2
```

#### 72. Edit Distance

```python3
class Solution:
    def minDistance1(self, word1: str, word2: str) -> int:
        mem = {}
        def ed(m, n):
            if (m, n) in mem:
                return mem[(m, n)]
            if m == -1:
                return n + 1
            if n == -1:
                return m + 1
            if word1[m] == word2[n]:
                mem[(m, n)] = ed(m-1, n-1)
            else:
                mem[(m, n)] = min(ed(m-1, n), ed(m-1, n-1), ed(m, n-1)) + 1
            return mem[(m, n)]
        m, n = len(word1) - 1, len(word2) - 1
        return ed(m, n)
    
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1) + 1, len(word2) + 1
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            dp[0][i] = i
        for i in range(m):
            dp[i][0] = i
        for i in range(1, m):
            for j in range(1, n):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]) + 1
        return dp[-1][-1]
```

### Subsequence

#### 300. Longest Increasing Subsequence

```python3
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1 for _ in range(len(nums))]
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

#### 1143. Longest Common Subsequence

```python3
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for _ in range(len(text1) + 1)] for _ in range(len(text2) + 1)]
        for i in range(1, len(text2) + 1):
            for j in range(1, len(text1) + 1):
                if text1[j-1] == text2[i-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```

#### 5. Longest Palindromic Substring

```python3
class Solution:
    def longestPalindrome(self, s):
        n = len(s)
        left, maxlen = 0, 1
        dp = [[False] * n for i in range(n)]
        
        # base case: one letter
        for i in range(n):
            dp[i][i] = True
        
        # base case: two same letter
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                left = i
                maxlen = 2
        
        # letter above two
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if dp[i + 1][j - 1] == True and s[i] == s[j]:
                    dp[i][j] = True
                    left = i
                    maxlen = length
        return s[left : left + maxlen]
```

### Knapsack problem

#### 322. Coin Change

```python3
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount+1] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i-coin] + 1)
        if dp[-1] == amount + 1:
            return -1
        return dp[-1]
```

### Stock Problem

#### 121. Best Time to Buy and Sell Stock (k=1)

```python3
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[0]*2 for _ in range(n)]
        dp[0][0], dp[0][1] = 0, -prices[0]
        for i in range(1, n):
            # dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1] + prices[i])
            # dp[i][1][1] = max(dp[i-1][1][1], dp[i-1][0][0] - prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])
        return dp[-1][0]
```

#### 122. Best Time to Buy and Sell Stock II (k=+infinity)

```python3
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[0]*2 for i in range(n)]
        dp[0][0], dp[0][1] = 0, -prices[0] 
        for i in range(1, n):
            # dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
            # dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[-1][0]
```

#### 714. Best Time to Buy and Sell Stock with Transaction Fee (k=+infinity with fee)

```python3
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp = [[0]*2 for i in range(n)]
        dp[0][0], dp[0][1] = 0, -prices[0]
        for i in range(1, n):
            # dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
            # dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[-1][0]
```

## Design Pattern

### Adapter

```python3
from abc import ABCMeta, abstractmethod

class Payment(metaclass=ABCMeta):
    @abstractmethod
    def pay(self, moeny):
        pass
      
class BankPayment(Payment):
    def cost(self, money):
        print(f"Bank pay: (money)})

class PaymentAdapter(Payment):
    def __init__(self, payment):
        self.payment = payment
     
    def pay(self, money):
        self.payment.cost(money)
    
if __name__ == "__main__":
    p = PaymentAdapter(BankPayment())
    p.pay(100)
```

### Proxy

```python3
from abc import ABCMeta, abstractmethod

class Subject(metaclass=ABCMeta):
    @abstractmethod
    def get_content(self):
        pass
        
    @abstractmethod
    def set_content(self, content):
        pass
    
class RealSubject(Subject):
		def __init__(self, filename):
		    self.filename = filename
		    with open(filename, "r", encoding="utf-8") as file:
		        self.content = file.read()
			
    def get_content(self):
        return self.content
     
    def set_content(self, content):
        with open(self.filename, "r", encoding="utf-8") as file:
            file.write(content)
        
class ProtectedProxy(Subject):
    def __init__(self, filename):
        self.subj = RealSubject(filename)
    
    def get_content(self):
        return self.subj.get_content()
    
    def set_content(self, content):
        raise PermissionError("No write permission!")
        
if __name__ == "__main__":
    subj = ProtectedProxy("test.txt")
    subj.get_content()
    subj.set_content('test')
```

### Factory

```python3
from abc import ABCMeta, abstractmethod

class Payment(metaclass=ABCMeta):
    @abstractmethod
    def pay(self, moeny):
        pass
      
class WechatPayment(Payment):
    def pay(self, money):
        print(f"Wechat pay: (money)})
        
class PaymentFactory(metaclass=ABCMeta):
    @abstractmethod
    def creat_payment(self):
        pass
  
 class WechatPayFactory(PaymentFactory):
     def create_payment(self):
         return WechatPayment()
     
 if __name__ == "__main__":
     pf = WechatPayFactory()
     p = pf.create_payment()
     p.pay(100)
```
