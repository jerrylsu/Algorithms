#include<vector>
#include<iostream>

class Solution{
    private:
        std::vector<std::vector<int>> res;
        std::vector<int> track;
    public:
        std::vector<std::vector<int>> permute(std::vector<int>& nums){
            if(nums.empty()){
                return res;
            }
            std::vector<bool> visited(nums.size(), false);
            dfs(nums, res, track, visited);
            return res;
        }

        void dfs(std::vector<int>& nums, \
                 std::vector<std::vector<int>>& res, \
                 std::vector<int>& track, \
                 std::vector<bool>& visited){
            if(track.size() == nums.size()){          // nums.size()层树
                res.push_back(track);
                return;
            }
            for(int i = 0; i < nums.size(); i++){     // nums.size()叉树
                if(!visited[i]){                      // 剪枝
                    track.push_back(nums[i]);
                    visited[i] = true;
                    dfs(nums, res, track, visited);
                    track.pop_back();
                    visited[i] = false;
                }
            }
        }
};

int main(){
    Solution s;
    std::vector<int> nums{1, 2, 3};
    std::vector<std::vector<int>> res;

    std::cout << "nums: ";
    for(int i = 0; i < nums.size(); i++){
        std::cout << nums[i] << " ";
    }
    std::cout << "\n";

    res = s.permute(nums);

    for(int i = 0; i < res.size(); i++){
        for(int j = 0; j < res[0].size(); j++){
            std::cout << res[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}