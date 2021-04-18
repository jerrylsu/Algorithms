class Solution:
    """https://leetcode.com/problems/longest-increasing-subsequence/discuss/1008117/Python-3-Brute-force(Recursive)-Recursion-%2B-Memo-DP-Binary_search
    """
    def lengthOfLIS(self, nums: list) -> int:
        """DP
        """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
