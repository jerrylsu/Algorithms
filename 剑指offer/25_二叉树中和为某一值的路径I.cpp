#include<vector>

using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : val(val), left(nullptr), right(nullptr) {}
};


class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        // 递归防守
        if(root == nullptr) return false;
        // 判断叶子结点
        if(root->left == nullptr && root->right == nullptr) {
            return sum == root->val;  
        }
        int new_sum = sum - root->val;
        return hasPathSum(root->left, new_sum) || hasPathSum(root->right, new_sum);
    }
};
