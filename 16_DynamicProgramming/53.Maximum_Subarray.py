class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [i for i in nums]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
        return max(dp)

    def maxSubArray(self, nums):
        if not nums: return
        dp, dp[0], res = [0, 0], nums[0], nums[0]
        for i in range(1, len(nums)):
            x, y = i % 2, (i - 1) % 2
            dp[x] = max(dp[y] + nums[i], nums[i])
            res = max(dp[x], res)
        return res
