
## 基本二分查找
### Classical Binary Search
- 数组元素无重复
```
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums or len(nums) == 0: return -1
        left, right = 0, len(nums) - 1
        while left + 1 < right:                  # 相邻即退出，防止某些问题造成死循环。
            mid = left + ((right - left) >> 1)    
            if nums[mid] == target:              # find target and return immediately due to no duplicated elements
                return mid
            elif target < nums[mid]:
                right = mid
            else:
                left = mid
        #if the target is not the mid, check the right and left
        if target == nums[left]: return left
        if target == nums[right]: return right
        return -1
```
模板四要素：
1. `left + 1 < right`
2. `left + ((right - left) >> 1)`
3. `nums[mid] ==`
4. `nums[left] A[right] ? target`

### Find First and Last Position of Target  ( lower & upper Bound )
- Target有重复，其他元素无重复。
```
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums: return [-1, -1]
        lower, upper = -1, -1
        
        # lower
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = left + ((right - left) >> 1)
            if target == nums[mid]:      # target有重复，找到target不立即退出。由于找下界，所以移动right指针。
                right = mid
            elif target < nums[mid]:
                right = mid
            else:
                left = mid
        if target == nums[right]: lower = right
        if target == nums[left]: lower = left
            
        # upper
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = left + ((right - left) >> 1)
            if target == nums[mid]:      # target有重复，找到target不立即退出。由于找上界，所以移动left指针。
                left = mid
            elif target < nums[mid]:
                right = mid
            else:
                left = mid
        if target == nums[left]: upper = left
        if target == nums[right]: upper = right
            
        return [lower, upper]
```

### 总结
1. 二分查找的本质是缩小搜索范围：**区间缩小 ===> 剩下两个下标 ===> 判断两个下标**
2. 时间复杂度
```
  T(n) = T(n/2) + O(1)
       = T(n/4) + O(1) + O(1)
       = T(n/8) + O(1) +O(1) +O(1)
       = T(1) + logn * O(1)
       = O(logn)
```

- `O(1)` 极少
- `O(logn)`几乎都是二分法
- `O(√n)` 几乎是分解质因数
- `O(n)` 高频
- `O(nlogn)` 一般都可能要排序
- `O(n2)` 数组，枚举，动态规划
- `O(n3)` 数组，枚举，动态规划
- `O(2^n)` 与组合有关的搜索 combination
- `O(n!)` 与排列有关的搜索 permutation

比`O(n)`更优的时间复杂度，几乎只能是`O(logn)`的二分法。经验之谈：**根据时间复杂度倒推算法是面试中的常用策略**