#include<iostream>
#include<vector>

using namespace std;

class Solution {
    public:
        vector<int> twoSum(vector<int>& nums, int target){
            vector<int> res;
            if(nums.empty()) return res;
            int sum, left = 0, right = nums.size() - 1;
            while(left < right){
                sum = nums[left] + nums[right];
                if(target == sum){
                    res.push_back(nums[left]);
                    res.push_back(nums[right]);
                    return res;
                }
                sum < target ? left++ : right--;
            }
            return res;
        }
};