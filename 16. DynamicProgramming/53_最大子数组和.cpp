#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<limits>

using namespace std;

class Solution {
    public:
        int maxSubArray(vector<int>& nums) {
            int n = nums.size();
            if(n == 0) return 0;
            vector<int> dp(n, 0);
            dp[0] = nums[0];
            int s_max = nums[0];
            for(int i = 1; i < n; ++i){
                dp[i] = max(nums[i], dp[i-1] + nums[i]);
                s_max = max(s_max, dp[i]);
            }
            return s_max;
        }
};

int main(){
    Solution solution;
    vector<int> nums{2, 4, -1, 7, 3, -3, 1};  // dp = [2, 6, 5, 12 ,15, 1,2 ,13]
    cout << solution.maxSubArray(nums) << endl;
    return 0;
} 
