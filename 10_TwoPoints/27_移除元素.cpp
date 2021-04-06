#include<vector>
#include<iostream>
using namespace std;
using std::vector;

class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int left = 0;
        int right = nums.size() - 1;
        while(left <= right){
            if(nums[left] == val){    // 相等的值放到到right区间
                swap(nums[left], nums[right]);
                right -= 1;
            }else                     // 不相等的放到left区间
            {
                left += 1;
            }
        }
        return left;
    }
};

int main(){
    vector<int> nums{3, 2, 2, 3};
    int val =3;
    Solution s;
    std::cout << s.removeElement(nums, val);
    
    return 0;
}