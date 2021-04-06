#include<iostream>
#include<algorithm>
#include<vector>

using namespace std;

class Solution {
public:
    void QuickSort(vector<int>& nums) {
        if(nums.size() < 2) return;
        quickSort(nums, 0, nums.size() - 1);
    }

    void quickSort(vector<int>& nums, int left, int right) {
        if(left < right) {
            auto p = partition(nums, left, right);
            quickSort(nums, left, p.first - 1);
            quickSort(nums, p.second + 1, right);
        }
    }

    pair<int, int> partition(vector<int>& nums, int left, int right) {
        int pivot = nums[right];
        int less = left - 1, more = right;
        while(left < more) {
            if(nums[left] < pivot)
                std::swap(nums[++less], nums[left++]);
            else if(nums[left] > pivot)
                std::swap(nums[left], nums[--more]);
            else
                left++;
        }
        std::swap(nums[--more], pivot);
        return make_pair(++less, more);
    }
};


int main() {
    Solution s;
    vector<int> nums{9, 7, 3, 5, 8, 9, 8};

    for(auto& ele : nums)
        cout << ele << " ";
    cout << endl;
    
    s.QuickSort(nums);
    
    for(auto& ele : nums)
        cout << ele << " ";
    cout << endl;

    return 0;
}
