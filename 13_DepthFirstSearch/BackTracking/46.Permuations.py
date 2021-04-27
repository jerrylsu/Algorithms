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
