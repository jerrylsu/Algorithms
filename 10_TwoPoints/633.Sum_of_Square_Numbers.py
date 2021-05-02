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
