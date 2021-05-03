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
