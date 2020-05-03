#include<vector>
#include<iostream>

// two points

class Solution{
    public:
        void merge(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n){
            int p1 = m - 1;
            int p2 = n - 1;
            int end = m + n - 1;
            while(p1 >= 0 && p2 >= 0){
                if(nums1[p1] > nums2[p2]){
                    nums1[end--] = nums1[p1--];
                }else{
                    nums1[end--] = nums2[p2--];
                }
            }
            
            while(p2 >= 0){
                nums1[end--] = nums2[p2--];
            }
        }
};