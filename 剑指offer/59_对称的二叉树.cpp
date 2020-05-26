#include<algorithm>

class TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : val(val), left(nullptr), right(nullptr {}
};


class Solution {
public:
    bool isSymmetrical(TreeNode* root){
        return root ? isSymmetrical(root->left, root->right) : true;
    }

    bool isSymmetrical(TreeNode* node1, TreeNode* node2){
        // case 1: 两个子树均为空
        if(node1 == nullptr && node2 == nullptr) return true;
        // case 2: 存在一个子树为空
        if(node1 == nullptr || node2 == nullptr) return false;
        // case 3：两个子树根结点的值不相等
        if(node1->val != node2->val) return false;
        // 递归处理
        return isSymmetrical(node1->left, node2->right) && helper(node1->right, node2->left);
    }
};
